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
            # Expected sequence length: 4 bbox + 1 confidence + 1 end = 6 tokens per frame
            self.tokens_per_frame = 6
        else:
            # For legacy mode
            self.bbox_vocab_size = self.BINS
            self.total_vocab = self.BINS + 2
            self.start_token = self.total_vocab - 2
            self.end_token = self.total_vocab - 1
            # Expected sequence length: 4 bbox + 1 end = 5 tokens per frame
            self.tokens_per_frame = 5

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

        # Dynamic vocabulary size adjustment
        self._dynamic_vocab_sizes = {}

    def __call__(self, data):
        """Main training forward pass with enhanced debugging"""
        try:
            print(f"Input data shapes:")
            print(f"  search_images: {data['search_images'].shape}")
            print(f"  search_anno: {data['search_anno'].shape}")
            print(f"  template_images: {data['template_images'].shape}")

            outputs, targets_dict = self.forward_pass(data)

            print(f"Forward pass outputs:")
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
            else:
                print(f"  outputs: {outputs.shape if hasattr(outputs, 'shape') else type(outputs)}")

            loss, status = self.compute_losses(outputs, targets_dict)
            return loss, status
        except Exception as e:
            print(f"Error in SeqTrackActor: {e}")
            print(f"Data shapes: search_anno: {data['search_anno'].shape}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            raise e

    def _update_dynamic_vocab_sizes(self, outputs):
        """Update vocabulary sizes based on actual model outputs"""
        if isinstance(outputs, dict):
            if 'bbox_logits' in outputs:
                actual_bbox_vocab = outputs['bbox_logits'].shape[-1]
                if actual_bbox_vocab != self.bbox_vocab_size:
                    print(f"Updating bbox vocab size from {self.bbox_vocab_size} to {actual_bbox_vocab}")
                    self._dynamic_vocab_sizes['bbox'] = actual_bbox_vocab

            if 'confidence_logits' in outputs:
                actual_conf_vocab = outputs['confidence_logits'].shape[-1]
                if actual_conf_vocab != self.CONF_BINS:
                    print(f"Updating confidence vocab size from {self.CONF_BINS} to {actual_conf_vocab}")
                    self._dynamic_vocab_sizes['confidence'] = actual_conf_vocab

    def _get_effective_vocab_size(self, vocab_type):
        """Get the effective vocabulary size, either configured or dynamically detected"""
        if vocab_type == 'bbox':
            return self._dynamic_vocab_sizes.get('bbox', self.bbox_vocab_size)
        elif vocab_type == 'confidence':
            return self._dynamic_vocab_sizes.get('confidence', self.CONF_BINS)
        else:
            return self.total_vocab

    def forward_pass(self, data):
        """Enhanced forward pass with confidence-aware processing"""
        n, b, _, _, _ = data['search_images'].shape
        print(f"Batch dimensions: n={n}, b={b}")

        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])
        search_list = search_img.split(b, dim=0)
        template_img = data['template_images'].view(-1, *data['template_images'].shape[2:])
        template_list = template_img.split(b, dim=0)

        # Encoder forward pass
        feature_xz = self.net(images_list=template_list + search_list, mode='encoder')

        # Process ground truth targets
        targets = data['search_anno'].permute(1, 0, 2).reshape(-1, data['search_anno'].shape[2])
        print(f"Targets shape after processing: {targets.shape}")

        # Handle different target formats
        if targets.shape[1] == 4:
            # Standard format: [x, y, w, h] - need to generate confidence
            bbox_coords = targets
            confidence_scores = self._generate_confidence_scores(bbox_coords, data)
            print(f"Generated confidence scores shape: {confidence_scores.shape}")
        elif targets.shape[1] == 5:
            # Extended format: [x, y, w, h, confidence] - use provided confidence
            bbox_coords = targets[:, :4]
            confidence_scores = targets[:, 4]
            print(f"Using provided confidence scores shape: {confidence_scores.shape}")
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

        print(f"Discretized tokens:")
        print(f"  bbox_tokens: {bbox_tokens.shape}")
        if confidence_tokens is not None:
            print(f"  confidence_tokens: {confidence_tokens.shape}")

        # Handle coordinate format
        if self.seq_format == 'whxy':
            bbox_tokens = bbox_tokens[:, [2, 3, 0, 1]]

        # Create input and target sequences
        batch_size = bbox_tokens.shape[0]
        print(f"Processing batch_size: {batch_size}")

        # Input sequence: [start] (for teacher forcing)
        start_tokens = torch.full((batch_size, 1), self.start_token, dtype=torch.int64, device=bbox_tokens.device)
        input_seqs = torch.cat([start_tokens, bbox_tokens], dim=1)

        # Reshape to match expected decoder input format
        expected_seq_length = b * n * self.tokens_per_frame
        actual_seq_length = input_seqs.numel()
        print(f"Sequence lengths - expected: {expected_seq_length}, actual: {actual_seq_length}")

        # Adjust sequence length if needed
        if actual_seq_length != expected_seq_length:
            print(f"Warning: Sequence length mismatch. Adjusting...")
            # Pad or truncate as needed
            if actual_seq_length < expected_seq_length:
                padding_needed = expected_seq_length - actual_seq_length
                padding = torch.zeros(padding_needed, dtype=torch.int64, device=bbox_tokens.device)
                input_seqs = torch.cat([input_seqs.flatten(), padding])
            else:
                input_seqs = input_seqs.flatten()[:expected_seq_length]
            input_seqs = input_seqs.view(b, n, -1).flatten(1)
        else:
            input_seqs = input_seqs.view(b, n, -1).flatten(1)

        print(f"Final input_seqs shape: {input_seqs.shape}")

        # Check if we have confidence-aware decoder
        has_confidence_decoder = (hasattr(self.net.decoder, 'confidence_enabled') and
                                  self.net.decoder.confidence_enabled and
                                  hasattr(self.net.decoder, 'forward_training'))

        if self.confidence_enabled and has_confidence_decoder:
            # Use new confidence-aware training mode
            try:
                # Prepare properly shaped tokens for new mode
                bbox_tokens_reshaped = bbox_tokens.view(b, n, -1).flatten(0, 1)
                confidence_tokens_reshaped = confidence_tokens.view(b, n, -1).flatten(0,
                                                                                      1) if confidence_tokens is not None else None

                print(f"Tokens for confidence-aware mode:")
                print(f"  bbox_tokens_reshaped: {bbox_tokens_reshaped.shape}")
                if confidence_tokens_reshaped is not None:
                    print(f"  confidence_tokens_reshaped: {confidence_tokens_reshaped.shape}")

                outputs = self.net(
                    xz=feature_xz,
                    seq=input_seqs,
                    mode="decoder",
                    bbox_tokens=bbox_tokens_reshaped,
                    confidence_tokens=confidence_tokens_reshaped,
                    is_training=True
                )

                # Update dynamic vocabulary sizes based on actual outputs
                self._update_dynamic_vocab_sizes(outputs)

                # Prepare targets for new mode
                targets_dict = {
                    'bbox_tokens': bbox_tokens_reshaped,
                    'confidence_tokens': confidence_tokens_reshaped
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

        print(f"Legacy mode target_seqs shape: {target_seqs.shape}")

        # Run decoder
        outputs = self.net(xz=feature_xz, seq=input_seqs, mode="decoder")
        print(f"Legacy mode raw outputs shape: {outputs.shape if hasattr(outputs, 'shape') else 'not tensor'}")

        # Reshape outputs with proper validation
        if not isinstance(outputs, dict):
            if hasattr(outputs, '__getitem__') and len(outputs) > 0:
                raw_output = outputs[-1]
            else:
                raw_output = outputs

            print(f"Raw output shape before reshaping: {raw_output.shape}")

            # Calculate expected output size
            expected_total_tokens = b * n * self.tokens_per_frame
            actual_output_size = raw_output.shape[0] if len(raw_output.shape) > 0 else raw_output.numel()

            print(f"Expected total tokens: {expected_total_tokens}")
            print(f"Actual output size: {actual_output_size}")

            # Validate and adjust if necessary
            if actual_output_size != expected_total_tokens:
                print(f"Warning: Output size mismatch. Adjusting...")
                if len(raw_output.shape) == 2:
                    # Output is already [tokens, vocab_size]
                    outputs = raw_output
                else:
                    # Need to reshape
                    if actual_output_size % self.total_vocab == 0:
                        outputs = raw_output.view(-1, self.total_vocab)
                    else:
                        # Emergency fallback: truncate or pad
                        target_size = (expected_total_tokens, self.total_vocab)
                        if raw_output.numel() >= target_size[0] * target_size[1]:
                            outputs = raw_output.flatten()[:target_size[0] * target_size[1]].view(target_size)
                        else:
                            # Pad with zeros
                            needed_elements = target_size[0] * target_size[1]
                            padded = torch.zeros(needed_elements, device=raw_output.device, dtype=raw_output.dtype)
                            padded[:raw_output.numel()] = raw_output.flatten()
                            outputs = padded.view(target_size)
            else:
                outputs = raw_output.reshape(-1, self.total_vocab)

        print(f"Final legacy outputs shape: {outputs.shape if hasattr(outputs, 'shape') else type(outputs)}")

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
        # Use dynamic vocab size for confidence
        effective_conf_bins = self._get_effective_vocab_size('confidence')
        confidence_tokens = (confidence_scores * (effective_conf_bins - 1)).long()
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
        """Compute losses in confidence-aware mode with proper shape validation"""
        losses = {}
        device = next(iter(outputs.values())).device

        # Get effective vocabulary sizes
        effective_bbox_vocab = self._get_effective_vocab_size('bbox')
        effective_conf_vocab = self._get_effective_vocab_size('confidence')

        # Bbox loss with shape validation
        if 'bbox_logits' in outputs and 'bbox_tokens' in targets_dict:
            bbox_logits = outputs['bbox_logits']
            bbox_targets = targets_dict['bbox_tokens']

            print(f"Bbox loss computation:")
            print(f"  bbox_logits shape: {bbox_logits.shape}")
            print(f"  bbox_targets shape: {bbox_targets.shape}")
            print(f"  effective_bbox_vocab: {effective_bbox_vocab}")

            # Handle different tensor shapes
            if len(bbox_logits.shape) == 3:
                # Shape: [batch, seq_len, vocab_size]
                batch_size, seq_len, vocab_size = bbox_logits.shape
                bbox_logits = bbox_logits.view(-1, vocab_size)
                bbox_targets = bbox_targets.view(-1)
                print(f"  Reshaped - logits: {bbox_logits.shape}, targets: {bbox_targets.shape}")

            # Validate vocabulary size
            actual_vocab_size = bbox_logits.shape[-1]
            if actual_vocab_size != effective_bbox_vocab:
                print(f"  Adjusting expected vocab size from {effective_bbox_vocab} to {actual_vocab_size}")
                effective_bbox_vocab = actual_vocab_size

            # Clamp targets to valid range
            bbox_targets = torch.clamp(bbox_targets, 0, effective_bbox_vocab - 1)

            # Remove padding tokens (assuming negative values are padding)
            valid_mask = (bbox_targets >= 0) & (bbox_targets < effective_bbox_vocab)
            if valid_mask.any():
                try:
                    bbox_loss = F.cross_entropy(bbox_logits[valid_mask], bbox_targets[valid_mask], reduction='mean')
                    losses['bbox_loss'] = bbox_loss
                    print(f"  Computed bbox_loss: {bbox_loss.item()}")
                except Exception as e:
                    print(f"  Bbox loss computation failed: {e}")
                    losses['bbox_loss'] = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                losses['bbox_loss'] = torch.tensor(0.0, device=device, requires_grad=True)
                print(f"  No valid bbox targets, using zero loss")

        # Confidence loss with shape validation
        if self.confidence_enabled and 'confidence_logits' in outputs and 'confidence_tokens' in targets_dict:
            conf_logits = outputs['confidence_logits']
            conf_targets = targets_dict['confidence_tokens']

            print(f"Confidence loss computation:")
            print(f"  conf_logits shape: {conf_logits.shape}")
            print(f"  conf_targets shape: {conf_targets.shape}")
            print(f"  effective_conf_vocab: {effective_conf_vocab}")

            # Handle different tensor shapes
            if len(conf_logits.shape) == 3:
                # Shape: [batch, seq_len, vocab_size]
                batch_size, seq_len, vocab_size = conf_logits.shape
                conf_logits = conf_logits.view(-1, vocab_size)
                conf_targets = conf_targets.view(-1)
                print(f"  Reshaped - logits: {conf_logits.shape}, targets: {conf_targets.shape}")

            # Validate vocabulary size
            actual_conf_vocab_size = conf_logits.shape[-1]
            if actual_conf_vocab_size != effective_conf_vocab:
                print(f"  Adjusting expected conf vocab size from {effective_conf_vocab} to {actual_conf_vocab_size}")
                effective_conf_vocab = actual_conf_vocab_size

            # Clamp targets to valid range
            conf_targets = torch.clamp(conf_targets, 0, effective_conf_vocab - 1)

            # Remove invalid confidence tokens
            valid_mask = (conf_targets >= 0) & (conf_targets < effective_conf_vocab)
            print(
                f"  Valid mask shape: {valid_mask.shape}, conf_targets range: [{conf_targets.min()}, {conf_targets.max()}]")

            if valid_mask.any():
                try:
                    if self.confidence_loss_type == 'focal':
                        conf_loss = self._focal_loss(conf_logits[valid_mask], conf_targets[valid_mask])
                    else:
                        conf_loss = F.cross_entropy(conf_logits[valid_mask], conf_targets[valid_mask], reduction='mean')
                    losses['confidence_loss'] = conf_loss
                    print(f"  Computed confidence_loss: {conf_loss.item()}")
                except Exception as e:
                    print(f"  Confidence loss computation failed: {e}")
                    print(
                        f"  conf_logits[valid_mask] shape: {conf_logits[valid_mask].shape if valid_mask.any() else 'empty'}")
                    print(
                        f"  conf_targets[valid_mask] shape: {conf_targets[valid_mask].shape if valid_mask.any() else 'empty'}")
                    losses['confidence_loss'] = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                losses['confidence_loss'] = torch.tensor(0.0, device=device, requires_grad=True)
                print(f"  No valid confidence targets, using zero loss")

        # Confidence-weighted bbox loss (advanced feature)
        if 'bbox_loss' in losses and 'confidence_logits' in outputs and losses['bbox_loss'].item() > 0:
            try:
                confidence_probs = F.softmax(outputs['confidence_logits'], dim=-1)
                # Use mean confidence as weight
                mean_confidence = torch.mean(confidence_probs, dim=1).mean()
                weighted_bbox_loss = losses['bbox_loss'] * (0.5 + 0.5 * mean_confidence)
                losses['weighted_bbox_loss'] = weighted_bbox_loss
                print(f"  Computed weighted_bbox_loss: {weighted_bbox_loss.item()}")
            except Exception as e:
                print(f"Failed to compute weighted bbox loss: {e}")
                losses['weighted_bbox_loss'] = losses['bbox_loss']

        # Combined loss
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if 'bbox_loss' in losses:
            total_loss = total_loss + self.ce_weight * losses['bbox_loss']
        if 'confidence_loss' in losses:
            total_loss = total_loss + self.conf_loss_weight * losses['confidence_loss']

        losses['total_loss'] = total_loss
        print(f"Total loss: {total_loss.item()}")

        # Compute evaluation metrics
        if return_status:
            status = self._compute_status_metrics(outputs, targets_dict, losses)
            return total_loss, status
        else:
            return total_loss

    def _compute_legacy_losses(self, outputs, targets_dict, return_status=True):
        """Compute losses in legacy mode with enhanced shape validation"""
        if 'target_seqs' not in targets_dict:
            raise ValueError("Legacy mode requires target_seqs in targets_dict")

        targets_seq = targets_dict['target_seqs']
        device = outputs.device

        print(f"Legacy loss computation:")
        print(f"  outputs shape: {outputs.shape}")
        print(f"  targets_seq shape: {targets_seq.shape}")

        # Validate sequence lengths match
        if outputs.shape[0] != targets_seq.shape[0]:
            print(f"Warning: Output/target sequence length mismatch: {outputs.shape[0]} vs {targets_seq.shape[0]}")
            # Adjust to minimum length
            min_length = min(outputs.shape[0], targets_seq.shape[0])
            outputs = outputs[:min_length]
            targets_seq = targets_seq[:min_length]
            print(f"  Adjusted to length: {min_length}")

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

        print(f"  After masking - outputs: {outputs_masked.shape}, targets: {targets_masked.shape}")

        if self.confidence_enabled:
            # Separate bbox and confidence tokens
            is_conf = targets_masked >= self.confidence_vocab_offset

            # Separate losses
            ce_loss = F.cross_entropy(outputs_masked, targets_masked, reduction='none')
            main_loss = ce_loss[~is_conf].mean() if (~is_conf).any() else torch.tensor(0.0, device=device)
            conf_loss = ce_loss[is_conf].mean() if is_conf.any() else torch.tensor(0.0, device=device)

            total_loss = self.ce_weight * main_loss + self.conf_loss_weight * conf_loss
            print(f"  main_loss: {main_loss.item()}, conf_loss: {conf_loss.item()}")
        else:
            # Standard cross-entropy loss
            total_loss = F.cross_entropy(outputs_masked, targets_masked, reduction='mean')
            main_loss = total_loss
            conf_loss = torch.tensor(0.0, device=device)
            print(f"  standard loss: {total_loss.item()}")

        # Compute IoU for evaluation
        try:
            iou = self._compute_iou_legacy(outputs, targets_seq)
            print(f"  IoU: {iou.item()}")
        except Exception as e:
            print(f"  Failed to compute IoU: {e}")
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
        """Compute IoU for legacy mode with enhanced error handling"""
        device = outputs.device

        try:
            # Get predictions
            effective_bbox_vocab = self._get_effective_vocab_size('bbox')
            outputs_softmax = F.softmax(outputs, dim=-1)[:, :effective_bbox_vocab]
            preds = outputs_softmax.argmax(-1)

            print(f"IoU computation:")
            print(f"  preds shape: {preds.shape}")
            print(f"  targets_seq shape: {targets_seq.shape}")
            print(f"  tokens_per_frame: {self.tokens_per_frame}")

            # Reshape based on confidence mode
            try:
                if self.confidence_enabled:
                    preds = preds.reshape(-1, self.tokens_per_frame)[:, :4]  # Take first 4 (bbox tokens)
                    targets = targets_seq.reshape(-1, self.tokens_per_frame)[:, :4]
                else:
                    preds = preds.reshape(-1, self.tokens_per_frame)[:, :4]  # Take first 4 (bbox tokens)
                    targets = targets_seq.reshape(-1, self.tokens_per_frame)[:, :4]

                print(f"  After reshape - preds: {preds.shape}, targets: {targets.shape}")
            except Exception as e:
                print(f"  Reshape failed: {e}")
                # Fallback: try to extract bbox tokens manually
                min_length = min(preds.shape[0], targets_seq.shape[0])
                if min_length >= 4:
                    preds = preds[:min_length // 4 * 4].reshape(-1, 4)
                    targets = targets_seq[:min_length // 4 * 4].reshape(-1, 4)
                else:
                    return torch.tensor(0.0, device=device)

            # Convert to normalized coordinates
            preds_normalized = preds.float() / (effective_bbox_vocab - 1)
            targets_normalized = targets.float() / (effective_bbox_vocab - 1)

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

            # Clamp to valid ranges
            boxes_pred = torch.clamp(boxes_pred, 0.0, 1.0)
            boxes_target = torch.clamp(boxes_target, 0.0, 1.0)

            # Compute IoU
            iou = box_iou(boxes_pred, boxes_target)[0].mean()
            return iou

        except Exception as e:
            print(f"IoU computation failed: {e}")
            return torch.tensor(0.0, device=device)

    def _focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for confidence prediction"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def _compute_status_metrics(self, outputs: Dict[str, torch.Tensor],
                                targets_dict: Dict[str, torch.Tensor],
                                losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute detailed status metrics for logging with error handling"""
        status = {}

        # Basic loss metrics
        for key in ['total_loss', 'bbox_loss', 'confidence_loss', 'weighted_bbox_loss']:
            if key in losses:
                status[f"Loss/{key.replace('_loss', '')}"] = losses[key].item()

        # Compute IoU if possible
        if 'bbox_logits' in outputs and 'bbox_tokens' in targets_dict:
            try:
                effective_bbox_vocab = self._get_effective_vocab_size('bbox')
                bbox_preds = torch.argmax(outputs['bbox_logits'], dim=-1)
                bbox_targets = targets_dict['bbox_tokens']

                # Ensure same shape
                if bbox_preds.shape != bbox_targets.shape:
                    min_size = min(bbox_preds.numel(), bbox_targets.numel()) // 4 * 4
                    if min_size > 0:
                        bbox_preds = bbox_preds.view(-1)[:min_size].view(-1, 4)
                        bbox_targets = bbox_targets.view(-1)[:min_size].view(-1, 4)
                    else:
                        status["IoU"] = 0.0
                        return status

                # Convert to normalized coordinates
                preds_norm = bbox_preds.float() / (effective_bbox_vocab - 1)
                targets_norm = bbox_targets.float() / (effective_bbox_vocab - 1)

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

                # Clamp and compute IoU
                boxes_pred = torch.clamp(boxes_pred, 0.0, 1.0)
                boxes_target = torch.clamp(boxes_target, 0.0, 1.0)

                iou = box_iou(boxes_pred, boxes_target)[0].mean()
                status["IoU"] = iou.item()
            except Exception as e:
                print(f"IoU computation in status failed: {e}")
                status["IoU"] = 0.0

        # Confidence-specific metrics
        if self.confidence_enabled and 'confidence_logits' in outputs and 'confidence_tokens' in targets_dict:
            try:
                effective_conf_vocab = self._get_effective_vocab_size('confidence')
                conf_preds = torch.argmax(outputs['confidence_logits'], dim=-1)
                conf_targets = targets_dict['confidence_tokens']

                # Flatten for comparison
                conf_preds_flat = conf_preds.view(-1)
                conf_targets_flat = conf_targets.view(-1)

                # Remove invalid targets
                valid_mask = (conf_targets_flat >= 0) & (conf_targets_flat < effective_conf_vocab)
                if valid_mask.any():
                    conf_accuracy = (conf_preds_flat[valid_mask] == conf_targets_flat[valid_mask]).float().mean()
                    status["Conf/accuracy"] = conf_accuracy.item()

                    # Mean predicted confidence
                    conf_probs = F.softmax(outputs['confidence_logits'], dim=-1)
                    bin_indices = torch.arange(effective_conf_vocab, device=conf_probs.device).float()
                    mean_conf = torch.sum(conf_probs * bin_indices, dim=-1).mean()
                    status["Conf/mean_pred"] = (mean_conf / (effective_conf_vocab - 1)).item()
                else:
                    status["Conf/accuracy"] = 0.0
                    status["Conf/mean_pred"] = 0.0
            except Exception as e:
                print(f"Confidence metrics computation failed: {e}")
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
