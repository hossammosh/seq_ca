from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_iou
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import numpy as np


class SeqTrackActor(BaseActor):
    """ Enhanced SeqTrack Actor with Confidence-Aware Training """

    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg

        # Vocabulary and confidence setup
        self.BINS = cfg.MODEL.BINS

        # Handle confidence bins configuration
        self.CONF_BINS = 100  # Default
        if hasattr(cfg.MODEL, 'CONFIDENCE_BINS'):
            self.CONF_BINS = cfg.MODEL.CONFIDENCE_BINS
        elif hasattr(cfg.MODEL, 'CONFIDENCE') and hasattr(cfg.MODEL.CONFIDENCE, 'BINS'):
            self.CONF_BINS = cfg.MODEL.CONFIDENCE.BINS

        # Handle confidence enabled flag
        self.confidence_enabled = True  # Default
        if hasattr(cfg.MODEL, 'CONFIDENCE_ENABLED'):
            self.confidence_enabled = cfg.MODEL.CONFIDENCE_ENABLED
        elif hasattr(cfg.MODEL, 'CONFIDENCE') and hasattr(cfg.MODEL.CONFIDENCE, 'ENABLED'):
            self.confidence_enabled = cfg.MODEL.CONFIDENCE.ENABLED

        # Sequence format
        self.seq_format = cfg.DATA.SEQ_FORMAT

        # Token setup based on confidence mode
        if self.confidence_enabled:
            # For confidence-aware mode: separate bbox and confidence vocabularies
            self.bbox_vocab_size = self.BINS
            self.confidence_vocab_offset = self.BINS
            self.total_vocab = self.BINS + self.CONF_BINS + 2  # +2 for start/end
            self.start_token = self.total_vocab - 2
            self.end_token = self.total_vocab - 1
        else:
            # For legacy mode
            self.bbox_vocab_size = self.BINS
            self.total_vocab = self.BINS + 2
            self.start_token = self.total_vocab - 2
            self.end_token = self.total_vocab - 1

        # Loss weights
        self.ce_weight = cfg.TRAIN.CE_WEIGHT

        # Handle confidence loss weight
        self.conf_loss_weight = 0.3  # Default
        if hasattr(cfg.TRAIN, 'CONFIDENCE_WEIGHT'):
            self.conf_loss_weight = cfg.TRAIN.CONFIDENCE_WEIGHT
        elif hasattr(cfg.TRAIN, 'CONFIDENCE') and hasattr(cfg.TRAIN.CONFIDENCE, 'WEIGHT'):
            self.conf_loss_weight = cfg.TRAIN.CONFIDENCE.WEIGHT

        # Advanced loss configuration
        self.confidence_loss_type = 'cross_entropy'
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        if hasattr(cfg.TRAIN, 'CONFIDENCE'):
            if hasattr(cfg.TRAIN.CONFIDENCE, 'LOSS_TYPE'):
                self.confidence_loss_type = cfg.TRAIN.CONFIDENCE.LOSS_TYPE
            if hasattr(cfg.TRAIN.CONFIDENCE, 'FOCAL_ALPHA'):
                self.focal_alpha = cfg.TRAIN.CONFIDENCE.FOCAL_ALPHA
            if hasattr(cfg.TRAIN.CONFIDENCE, 'FOCAL_GAMMA'):
                self.focal_gamma = cfg.TRAIN.CONFIDENCE.FOCAL_GAMMA

    def __call__(self, data):
        """Main training forward pass"""
        try:
            outputs, targets_dict = self.forward_pass(data)
            loss, status = self.compute_losses(outputs, targets_dict)
            return loss, status
        except Exception as e:
            print(f"Error in SeqTrackActor: {e}")
            print(f"Data shapes: search_anno: {data['search_anno'].shape}")
            raise e

    def forward_pass(self, data):
        """Enhanced forward pass with confidence-aware processing"""
        n, b, _, _, _ = data['search_images'].shape
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])
        search_list = search_img.split(b, dim=0)
        template_img = data['template_images'].view(-1, *data['template_images'].shape[2:])
        template_list = template_img.split(b, dim=0)

        # Encoder forward pass
        feature_xz = self.net(images_list=template_list + search_list, mode='encoder')

        # Process ground truth targets
        targets = data['search_anno'].permute(1, 0, 2).reshape(-1, data['search_anno'].shape[2])

        # Check the actual shape of targets
        print(f"Targets shape: {targets.shape}")  # Debug print

        # Handle different target formats
        if targets.shape[1] == 4:
            # Standard format: [x, y, w, h] - need to generate confidence
            bbox_coords = targets
            confidence_scores = self._generate_confidence_scores(bbox_coords, data)
        elif targets.shape[1] == 5:
            # Extended format: [x, y, w, h, confidence] - use provided confidence
            bbox_coords = targets[:, :4]
            confidence_scores = targets[:, 4]
        else:
            raise ValueError(f"Unexpected target format with {targets.shape[1]} columns. Expected 4 or 5.")

        # Convert bbox coordinates
        bbox_coords = box_xywh_to_xyxy(bbox_coords)
        bbox_coords = torch.clamp(bbox_coords, min=0.0, max=1.0)

        # Convert to the required coordinate format
        if self.seq_format not in ['corner', 'xyxy']:
            bbox_coords = box_xyxy_to_cxcywh(bbox_coords)

        # Discretize coordinates and confidence
        bbox_tokens = (bbox_coords * (self.bbox_vocab_size - 1)).long()
        confidence_tokens = self._discretize_confidence(confidence_scores) if self.confidence_enabled else None

        # Handle coordinate format
        if self.seq_format == 'whxy':
            bbox_tokens = bbox_tokens[:, [2, 3, 0, 1]]

        # Create input and target sequences
        batch_size = bbox_tokens.shape[0]

        # Input sequence: [start] (for teacher forcing)
        start_tokens = torch.full((batch_size, 1), self.start_token, dtype=torch.int64, device=bbox_tokens.device)
        input_seqs = torch.cat([start_tokens, bbox_tokens], dim=1)
        input_seqs = input_seqs.view(b, n, -1).flatten(1)

        # Check if we have confidence-aware decoder
        has_confidence_decoder = (hasattr(self.net.decoder, 'confidence_enabled') and
                                  self.net.decoder.confidence_enabled and
                                  hasattr(self.net.decoder, 'forward_training'))

        if self.confidence_enabled and has_confidence_decoder:
            # Use new confidence-aware training mode
            try:
                outputs = self.net(
                    xz=feature_xz,
                    seq=input_seqs,
                    mode="decoder",
                    bbox_tokens=bbox_tokens.view(b, n, -1).flatten(0, 1),
                    confidence_tokens=confidence_tokens.view(b, n, -1).flatten(0, 1),
                    is_training=True
                )

                # Prepare targets for new mode
                targets_dict = {
                    'bbox_tokens': bbox_tokens.view(b, n, -1).flatten(0, 1),
                    'confidence_tokens': confidence_tokens.view(b, n, -1).flatten(0, 1)
                }

            except Exception as e:
                print(f"Failed to use confidence-aware mode: {e}")
                print("Falling back to legacy mode...")
                # Fallback to legacy mode
                outputs, targets_dict = self._legacy_forward(
                    feature_xz, input_seqs, bbox_tokens, confidence_tokens, b, n
                )
        else:
            # Use legacy mode
            outputs, targets_dict = self._legacy_forward(
                feature_xz, input_seqs, bbox_tokens, confidence_tokens, b, n
            )

        return outputs, targets_dict

    def _legacy_forward(self, feature_xz, input_seqs, bbox_tokens, confidence_tokens, b, n):
        """Legacy forward pass for backward compatibility"""
        batch_size = bbox_tokens.shape[0]
        device = bbox_tokens.device

        # Create target sequence based on confidence mode
        if self.confidence_enabled and confidence_tokens is not None:
            # Create combined target sequence: [bbox, confidence, end]
            end_tokens = torch.full((batch_size, 1), self.end_token, dtype=torch.int64, device=device)
            adjusted_confidence = confidence_tokens + self.confidence_vocab_offset
            target_seqs = torch.cat([bbox_tokens, adjusted_confidence, end_tokens], dim=1)
            target_seqs = target_seqs.view(b, n, -1).flatten().long()
        else:
            # Standard target sequence: [bbox, end]
            end_tokens = torch.full((batch_size, 1), self.end_token, dtype=torch.int64, device=device)
            target_seqs = torch.cat([bbox_tokens, end_tokens], dim=1)
            target_seqs = target_seqs.view(b, n, -1).flatten().long()

        # Run decoder
        outputs = self.net(xz=feature_xz, seq=input_seqs, mode="decoder")

        # Reshape outputs
        if not isinstance(outputs, dict):
            if self.confidence_enabled:
                outputs = outputs[-1].reshape(-1, self.total_vocab)
            else:
                outputs = outputs[-1].reshape(-1, self.total_vocab)

        targets_dict = {'target_seqs': target_seqs}
        return outputs, targets_dict

    def _generate_confidence_scores(self, bbox_coords: torch.Tensor, data: Dict[str, Any]) -> torch.Tensor:
        """Generate confidence scores based on bbox properties and tracking context"""
        batch_size = bbox_coords.shape[0]
        device = bbox_coords.device

        # Convert to center format for analysis
        if self.seq_format == 'corner':
            # bbox_coords is in xyxy format
            centers = (bbox_coords[:, :2] + bbox_coords[:, 2:]) / 2.0
            sizes = bbox_coords[:, 2:] - bbox_coords[:, :2]
        else:
            # bbox_coords is in cxcywh format
            centers = bbox_coords[:, :2]
            sizes = bbox_coords[:, 2:]

        # Size-based confidence (prefer medium-sized objects)
        bbox_areas = sizes[:, 0] * sizes[:, 1]  # width * height

        # Optimal area around 10% of image (0.1), with tolerance
        optimal_area = 0.1
        area_diff = torch.abs(bbox_areas - optimal_area)
        size_confidence = torch.exp(-5.0 * area_diff)  # Exponential decay from optimal size
        size_confidence = torch.clamp(size_confidence, 0.2, 1.0)

        # Position-based confidence (prefer centered objects)
        center_x, center_y = centers[:, 0], centers[:, 1]
        distance_from_center = torch.sqrt((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2)
        center_confidence = torch.exp(-2.0 * distance_from_center)  # Exponential decay from center
        center_confidence = torch.clamp(center_confidence, 0.3, 1.0)

        # Aspect ratio confidence (prefer reasonable aspect ratios)
        width, height = sizes[:, 0], sizes[:, 1]
        aspect_ratio = width / (height + 1e-6)
        # Prefer aspect ratios between 0.5 and 2.0
        aspect_penalty = torch.abs(torch.log(aspect_ratio + 1e-6))
        aspect_confidence = torch.exp(-aspect_penalty)
        aspect_confidence = torch.clamp(aspect_confidence, 0.4, 1.0)

        # Size consistency confidence (avoid very small or very large objects)
        min_size = torch.min(width, height)
        max_size = torch.max(width, height)
        size_ratio = min_size / (max_size + 1e-6)
        size_consistency = 0.5 + 0.5 * size_ratio  # Reward square-ish objects

        # Combine different confidence factors with weights
        confidence_scores = (
                0.3 * size_confidence +
                0.25 * center_confidence +
                0.2 * aspect_confidence +
                0.25 * size_consistency
        )

        # Add controlled noise for robustness during training
        if self.training:
            noise = torch.randn_like(confidence_scores) * 0.03  # Small noise
            confidence_scores = confidence_scores + noise

        # Final clamping
        confidence_scores = torch.clamp(confidence_scores, 0.1, 0.95)

        return confidence_scores

    def _discretize_confidence(self, confidence_scores: torch.Tensor) -> torch.Tensor:
        """Discretize continuous confidence scores into bins"""
        if confidence_scores is None:
            return None

        confidence_scores = torch.clamp(confidence_scores, 0.0, 1.0)
        confidence_tokens = (confidence_scores * (self.CONF_BINS - 1)).long()
        return confidence_tokens.unsqueeze(1)

    def compute_losses(self, outputs, targets_dict, return_status=True):
        """Enhanced loss computation with confidence-aware components"""
        if isinstance(outputs, dict):
            # New confidence-aware mode
            return self._compute_confidence_aware_losses(outputs, targets_dict, return_status)
        else:
            # Legacy mode
            return self._compute_legacy_losses(outputs, targets_dict, return_status)

    def _compute_confidence_aware_losses(self, outputs: Dict[str, torch.Tensor],
                                         targets_dict: Dict[str, torch.Tensor],
                                         return_status: bool = True):
        """Compute losses in confidence-aware mode"""
        losses = {}
        device = next(iter(outputs.values())).device

        # Bbox loss
        if 'bbox_logits' in outputs and 'bbox_tokens' in targets_dict:
            bbox_logits = outputs['bbox_logits'].view(-1, self.bbox_vocab_size)
            bbox_targets = targets_dict['bbox_tokens'].view(-1)

            # Remove padding tokens (assuming 0 is padding)
            valid_mask = bbox_targets >= 0
            if valid_mask.any():
                bbox_loss = F.cross_entropy(bbox_logits[valid_mask], bbox_targets[valid_mask], reduction='mean')
                losses['bbox_loss'] = bbox_loss
            else:
                losses['bbox_loss'] = torch.tensor(0.0, device=device, requires_grad=True)

        # Confidence loss
        if self.confidence_enabled and 'confidence_logits' in outputs and 'confidence_tokens' in targets_dict:
            conf_logits = outputs['confidence_logits'].view(-1, self.CONF_BINS)
            conf_targets = targets_dict['confidence_tokens'].view(-1)

            # Remove invalid confidence tokens
            valid_mask = (conf_targets >= 0) & (conf_targets < self.CONF_BINS)
            if valid_mask.any():
                if self.confidence_loss_type == 'focal':
                    conf_loss = self._focal_loss(conf_logits[valid_mask], conf_targets[valid_mask])
                else:
                    conf_loss = F.cross_entropy(conf_logits[valid_mask], conf_targets[valid_mask], reduction='mean')
                losses['confidence_loss'] = conf_loss
            else:
                losses['confidence_loss'] = torch.tensor(0.0, device=device, requires_grad=True)

        # Confidence-weighted bbox loss (advanced feature)
        if 'bbox_loss' in losses and 'confidence_logits' in outputs and losses['bbox_loss'].item() > 0:
            try:
                confidence_probs = F.softmax(outputs['confidence_logits'], dim=-1)
                # Use mean confidence as weight
                mean_confidence = torch.mean(confidence_probs, dim=1).mean()
                weighted_bbox_loss = losses['bbox_loss'] * (0.5 + 0.5 * mean_confidence)
                losses['weighted_bbox_loss'] = weighted_bbox_loss
            except:
                losses['weighted_bbox_loss'] = losses['bbox_loss']

        # Combined loss
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if 'bbox_loss' in losses:
            total_loss = total_loss + self.ce_weight * losses['bbox_loss']
        if 'confidence_loss' in losses:
            total_loss = total_loss + self.conf_loss_weight * losses['confidence_loss']

        losses['total_loss'] = total_loss

        # Compute evaluation metrics
        if return_status:
            status = self._compute_status_metrics(outputs, targets_dict, losses)
            return total_loss, status
        else:
            return total_loss

    def _compute_legacy_losses(self, outputs, targets_dict, return_status=True):
        """Compute losses in legacy mode"""
        if 'target_seqs' not in targets_dict:
            raise ValueError("Legacy mode requires target_seqs in targets_dict")

        targets_seq = targets_dict['target_seqs']
        device = outputs.device

        # Apply padding mask
        pad_mask = targets_seq != 0

        if not pad_mask.any():
            # Handle edge case where all targets are padding
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if return_status:
                status = {"Loss/total": 0.0, "IoU": 0.0, "Loss/main": 0.0, "Loss/conf": 0.0}
                return total_loss, status
            return total_loss

        # Apply mask
        outputs_masked = outputs[pad_mask]
        targets_masked = targets_seq[pad_mask]

        if self.confidence_enabled:
            # Separate bbox and confidence tokens
            is_conf = targets_masked >= self.confidence_vocab_offset

            # Separate losses
            ce_loss = F.cross_entropy(outputs_masked, targets_masked, reduction='none')
            main_loss = ce_loss[~is_conf].mean() if (~is_conf).any() else torch.tensor(0.0, device=device)
            conf_loss = ce_loss[is_conf].mean() if is_conf.any() else torch.tensor(0.0, device=device)

            total_loss = self.ce_weight * main_loss + self.conf_loss_weight * conf_loss
        else:
            # Standard cross-entropy loss
            total_loss = F.cross_entropy(outputs_masked, targets_masked, reduction='mean')
            main_loss = total_loss
            conf_loss = torch.tensor(0.0, device=device)

        # Compute IoU for evaluation
        try:
            iou = self._compute_iou_legacy(outputs, targets_seq)
        except:
            iou = torch.tensor(0.0, device=device)

        if return_status:
            status = {
                "Loss/total": total_loss.item(),
                "IoU": iou.item(),
                "Loss/main": main_loss.item(),
                "Loss/conf": conf_loss.item() if self.confidence_enabled else 0.0
            }
            return total_loss, status
        else:
            return total_loss

    def _compute_iou_legacy(self, outputs, targets_seq):
        """Compute IoU for legacy mode"""
        device = outputs.device

        # Get predictions
        outputs_softmax = F.softmax(outputs, dim=-1)[:, :self.bbox_vocab_size]
        preds = outputs_softmax.argmax(-1)

        # Reshape based on confidence mode
        if self.confidence_enabled:
            try:
                preds = preds.reshape(-1, 6)[:, :4]  # [x,y,w,h,c,end] -> [x,y,w,h]
                targets = targets_seq.reshape(-1, 6)[:, :4]
            except:
                # Fallback if reshape fails
                return torch.tensor(0.0, device=device)
        else:
            try:
                preds = preds.reshape(-1, 5)[:, :4]  # [x,y,w,h,end] -> [x,y,w,h]
                targets = targets_seq.reshape(-1, 5)[:, :4]
            except:
                return torch.tensor(0.0, device=device)

        # Convert to normalized coordinates
        preds_normalized = preds.float() / (self.bbox_vocab_size - 1)
        targets_normalized = targets.float() / (self.bbox_vocab_size - 1)

        # Handle coordinate format
        if self.seq_format == 'whxy':
            preds_normalized = preds_normalized[:, [2, 3, 0, 1]]  # whxy -> xywh
            targets_normalized = targets_normalized[:, [2, 3, 0, 1]]

        # Convert to xyxy for IoU computation
        if self.seq_format != 'corner':
            boxes_pred = box_cxcywh_to_xyxy(preds_normalized)
            boxes_target = box_cxcywh_to_xyxy(targets_normalized)
        else:
            boxes_pred = preds_normalized
            boxes_target = targets_normalized

        # Compute IoU
        iou = box_iou(boxes_pred, boxes_target)[0].mean()
        return iou

    def _focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for confidence prediction"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def _compute_status_metrics(self, outputs: Dict[str, torch.Tensor],
                                targets_dict: Dict[str, torch.Tensor],
                                losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute detailed status metrics for logging"""
        status = {}

        # Basic loss metrics
        for key in ['total_loss', 'bbox_loss', 'confidence_loss', 'weighted_bbox_loss']:
            if key in losses:
                status[f"Loss/{key.replace('_loss', '')}"] = losses[key].item()

        # Compute IoU if possible
        if 'bbox_logits' in outputs and 'bbox_tokens' in targets_dict:
            try:
                bbox_preds = torch.argmax(outputs['bbox_logits'], dim=-1)
                bbox_targets = targets_dict['bbox_tokens']

                # Ensure same shape
                if bbox_preds.shape != bbox_targets.shape:
                    min_size = min(bbox_preds.numel(), bbox_targets.numel()) // 4 * 4
                    bbox_preds = bbox_preds.view(-1)[:min_size].view(-1, 4)
                    bbox_targets = bbox_targets.view(-1)[:min_size].view(-1, 4)

                # Convert to normalized coordinates
                preds_norm = bbox_preds.float() / (self.bbox_vocab_size - 1)
                targets_norm = bbox_targets.float() / (self.bbox_vocab_size - 1)

                # Handle coordinate format
                if self.seq_format == 'whxy':
                    preds_norm = preds_norm[:, [2, 3, 0, 1]]
                    targets_norm = targets_norm[:, [2, 3, 0, 1]]

                # Convert to xyxy and compute IoU
                if self.seq_format != 'corner':
                    boxes_pred = box_cxcywh_to_xyxy(preds_norm)
                    boxes_target = box_cxcywh_to_xyxy(targets_norm)
                else:
                    boxes_pred = preds_norm
                    boxes_target = targets_norm

                iou = box_iou(boxes_pred, boxes_target)[0].mean()
                status["IoU"] = iou.item()
            except Exception as e:
                status["IoU"] = 0.0

        # Confidence-specific metrics
        if self.confidence_enabled and 'confidence_logits' in outputs and 'confidence_tokens' in targets_dict:
            try:
                conf_preds = torch.argmax(outputs['confidence_logits'], dim=-1)
                conf_targets = targets_dict['confidence_tokens']

                # Flatten for comparison
                conf_preds_flat = conf_preds.view(-1)
                conf_targets_flat = conf_targets.view(-1)

                # Remove invalid targets
                valid_mask = (conf_targets_flat >= 0) & (conf_targets_flat < self.CONF_BINS)
                if valid_mask.any():
                    conf_accuracy = (conf_preds_flat[valid_mask] == conf_targets_flat[valid_mask]).float().mean()
                    status["Conf/accuracy"] = conf_accuracy.item()

                    # Mean predicted confidence
                    conf_probs = F.softmax(outputs['confidence_logits'], dim=-1)
                    bin_indices = torch.arange(self.CONF_BINS, device=conf_probs.device).float()
                    mean_conf = torch.sum(conf_probs * bin_indices, dim=-1).mean()
                    status["Conf/mean_pred"] = (mean_conf / (self.CONF_BINS - 1)).item()
                else:
                    status["Conf/accuracy"] = 0.0
                    status["Conf/mean_pred"] = 0.0
            except Exception as e:
                status["Conf/accuracy"] = 0.0
                status["Conf/mean_pred"] = 0.0

        return status

    def to(self, device):
        """Move actor to device"""
        self.net.to(device)
        if hasattr(self, 'objective') and self.objective is not None:
            if isinstance(self.objective, dict):
                for obj in self.objective.values():
                    if hasattr(obj, 'to'):
                        obj.to(device)
            else:
                if hasattr(self.objective, 'to'):
                    self.objective.to(device)

    @property
    def training(self):
        """Check if the network is in training mode"""
        return self.net.training if hasattr(self.net, 'training') else True
