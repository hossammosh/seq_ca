import copy
from typing import Optional, Dict, Tuple, Any
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import math


class DecoderEmbeddings(nn.Module):
    """Enhanced embeddings with type and positional encoding"""

    def __init__(self, vocab_size, hidden_dim, max_position_embeddings, dropout, confidence_enabled=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.confidence_enabled = confidence_enabled

        # Token embeddings for the full vocabulary (bbox + confidence + special tokens)
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim)

        # Positional embeddings for sequence positions
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)

        # Type embeddings to distinguish bbox vs confidence tokens
        if confidence_enabled:
            self.type_embeddings = nn.Embedding(3, hidden_dim)  # [bbox, confidence, special]

        self.LayerNorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, position_ids=None, type_ids=None):
        """
        Args:
            input_ids: [batch, seq_len] - token ids
            position_ids: [batch, seq_len] - position ids (optional)
            type_ids: [batch, seq_len] - type ids (optional)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        input_embeds = self.word_embeddings(input_ids)

        # Position embeddings
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)

        # Type embeddings
        type_embeds = torch.zeros_like(input_embeds)
        if self.confidence_enabled and type_ids is not None:
            type_embeds = self.type_embeddings(type_ids)

        # Combine embeddings
        embeddings = input_embeds + position_embeds + type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class ConfidenceAwareSeqTrackDecoder(nn.Module):
    """Enhanced SeqTrack Decoder with Confidence-Aware Sequence Modeling"""

    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, activation="relu", normalize_before=False,
                 return_intermediate_dec=False, bins=1000, confidence_bins=100,
                 num_frames=9, confidence_enabled=True):
        super().__init__()

        # =============================================================================
        # CORE PARAMETERS AND INITIALIZATION
        # =============================================================================

        self.d_model = d_model
        self.nhead = nhead
        self.bins = bins
        self.confidence_bins = confidence_bins
        self.confidence_enabled = confidence_enabled
        self.num_coordinates = 4  # [x, y, w, h]

        # Vocabulary setup
        if confidence_enabled:
            self.bbox_vocab_size = bins
            self.confidence_vocab_offset = bins
            self.total_bins = bins + confidence_bins + 2  # +2 for start/end tokens
            self.sequence_length = 5  # [x, y, w, h, c]
        else:
            self.bbox_vocab_size = bins
            self.total_bins = bins + 2  # +2 for start/end tokens
            self.sequence_length = 4  # [x, y, w, h]

        # Special tokens
        self.start_token_id = self.total_bins - 2
        self.end_token_id = self.total_bins - 1
        self.pad_token_id = 0

        # Max position embeddings
        max_position_embeddings = (self.sequence_length + 2) * num_frames  # +2 for start/end

        # =============================================================================
        # EMBEDDING LAYERS
        # =============================================================================

        self.embedding = DecoderEmbeddings(
            vocab_size=self.total_bins,
            hidden_dim=d_model,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            confidence_enabled=confidence_enabled
        )

        # =============================================================================
        # TRANSFORMER DECODER ARCHITECTURE
        # =============================================================================

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.body = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm,
            return_intermediate=return_intermediate_dec
        )

        # =============================================================================
        # OUTPUT HEADS - DUAL HEAD ARCHITECTURE
        # =============================================================================

        # Shared feature extraction
        self.shared_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Bounding box prediction head
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.bbox_vocab_size)  # Only bbox bins
        )

        # Confidence prediction head (only if confidence is enabled)
        if confidence_enabled:
            self.confidence_head = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, confidence_bins)  # Only confidence bins
            )

            # Confidence-specific components
            self.confidence_weighting = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Sigmoid()
            )

            self.confidence_gate = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )

            self.confidence_calibration = nn.Linear(confidence_bins, confidence_bins)

        # Cross-attention projection for visual features
        self.cross_attn_proj = nn.Linear(d_model, d_model)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize model parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal attention mask for autoregressive generation"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def prepare_input_sequence(self, bbox_tokens: torch.Tensor,
                               confidence_tokens: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Prepare input sequence with proper embeddings

        Args:
            bbox_tokens: [batch, 4] - tokenized bbox coordinates [x, y, w, h]
            confidence_tokens: [batch, 1] - tokenized confidence values (optional)

        Returns:
            Dictionary containing input_ids, position_ids, and type_ids
        """
        batch_size = bbox_tokens.size(0)
        device = bbox_tokens.device

        # Create input sequence: [start, x, y, w, h, confidence, end] or [start, x, y, w, h, end]
        if self.confidence_enabled and confidence_tokens is not None:
            # Adjust confidence tokens to proper vocabulary range
            adjusted_confidence = confidence_tokens + self.confidence_vocab_offset
            input_ids = torch.cat([
                torch.full((batch_size, 1), self.start_token_id, device=device),  # start token
                bbox_tokens,  # [x, y, w, h]
                adjusted_confidence,  # confidence
                torch.full((batch_size, 1), self.end_token_id, device=device)  # end token
            ], dim=1)  # [batch, 7]
            seq_len = 7

            # Type IDs: [special, bbox, bbox, bbox, bbox, confidence, special]
            type_ids = torch.tensor([2, 0, 0, 0, 0, 1, 2], device=device).unsqueeze(0).expand(batch_size, -1)
        else:
            # Original sequence without confidence
            input_ids = torch.cat([
                torch.full((batch_size, 1), self.start_token_id, device=device),  # start token
                bbox_tokens,  # [x, y, w, h]
                torch.full((batch_size, 1), self.end_token_id, device=device)  # end token
            ], dim=1)  # [batch, 6]
            seq_len = 6

            # Type IDs: [special, bbox, bbox, bbox, bbox, special]
            type_ids = torch.tensor([2, 0, 0, 0, 0, 2], device=device).unsqueeze(0).expand(batch_size, -1)

        # Position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        return {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'type_ids': type_ids if self.confidence_enabled else None
        }

    def forward(self, src, pos_embed, seq, bbox_tokens=None, confidence_tokens=None, is_training=True):
        """
        Forward pass of confidence-aware decoder

        Args:
            src: [n, batch, d_model] - encoded visual features
            pos_embed: [n, batch, d_model] - positional embeddings
            seq: [batch, seq_len] - input sequence (for compatibility)
            bbox_tokens: [batch, 4] - bbox tokens (for training)
            confidence_tokens: [batch, 1] - confidence tokens (for training)
            is_training: bool - training or inference mode
        """
        if is_training and bbox_tokens is not None:
            return self.forward_training(src, pos_embed, bbox_tokens, confidence_tokens)
        else:
            return self.forward_inference_legacy(src, pos_embed, seq)

    def forward_training(self, src, pos_embed, bbox_tokens, confidence_tokens=None):
        """Training forward pass with teacher forcing"""
        n, batch_size, c = src.shape
        device = src.device

        # Prepare input sequence
        seq_data = self.prepare_input_sequence(bbox_tokens, confidence_tokens)

        # Get embeddings
        tgt = self.embedding(
            input_ids=seq_data['input_ids'],
            position_ids=seq_data['position_ids'],
            type_ids=seq_data['type_ids']
        ).permute(1, 0, 2)  # [seq_len, batch, hidden_dim]

        # Create causal mask
        seq_len = tgt.size(0)
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(device)

        # Transformer decoder forward pass
        hs = self.body(tgt, src, pos=pos_embed, query_pos=None, tgt_mask=tgt_mask, memory_mask=None)

        # Get the last layer output: [seq_len, batch, hidden_dim]
        decoder_output = hs[-1] if isinstance(hs, tuple) or len(hs.shape) == 4 else hs
        decoder_output = decoder_output.transpose(0, 1)  # [batch, seq_len, hidden_dim]

        # Shared feature processing
        shared_features = self.shared_head(decoder_output)  # [batch, seq_len, hidden_dim]

        outputs = {}

        # Bbox predictions for positions 1-4 (x, y, w, h) - skip start token
        bbox_features = shared_features[:, 1:5, :]  # [batch, 4, hidden_dim]
        bbox_logits = self.bbox_head(bbox_features)  # [batch, 4, bbox_vocab_size]
        outputs['bbox_logits'] = bbox_logits

        # Confidence predictions (if enabled) for position 5
        if self.confidence_enabled:
            confidence_features = shared_features[:, 5:6, :]  # [batch, 1, hidden_dim]
            confidence_logits = self.confidence_head(confidence_features)  # [batch, 1, confidence_bins]

            # Apply confidence calibration
            confidence_logits = self.confidence_calibration(confidence_logits)
            outputs['confidence_logits'] = confidence_logits

            # Combined logits for unified loss computation
            combined_logits = self._combine_logits(bbox_logits, confidence_logits)
            outputs['combined_logits'] = combined_logits

        return outputs

    def forward_inference_legacy(self, src, pos_embed, seq):
        """Legacy inference mode for compatibility"""
        n, bs, c = src.shape
        tgt = self.embedding(seq).permute(1, 0, 2)

        # Use position embeddings from the embedding layer
        query_embed = self.embedding.position_embeddings.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt_mask = generate_square_subsequent_mask(len(tgt)).to(tgt.device)

        hs = self.body(tgt, src, pos=pos_embed, query_pos=query_embed[:len(tgt)],
                       tgt_mask=tgt_mask, memory_mask=None)
        return hs.transpose(1, 2) if len(hs.shape) == 3 else hs[-1].transpose(1, 2)

    def _combine_logits(self, bbox_logits: torch.Tensor, confidence_logits: torch.Tensor) -> torch.Tensor:
        """
        Combine bbox and confidence logits into unified vocabulary space

        Args:
            bbox_logits: [batch, 4, bbox_vocab_size]
            confidence_logits: [batch, 1, confidence_bins]

        Returns:
            combined_logits: [batch, 5, total_vocab_size]
        """
        batch_size = bbox_logits.size(0)
        device = bbox_logits.device

        # Create combined logits tensor
        combined_logits = torch.full((batch_size, 5, self.total_bins), float('-inf'), device=device)

        # Fill bbox positions (0-3) with bbox logits
        combined_logits[:, :4, :self.bbox_vocab_size] = bbox_logits

        # Fill confidence position (4) with confidence logits
        combined_logits[:, 4,
        self.confidence_vocab_offset:self.confidence_vocab_offset + self.confidence_bins] = confidence_logits.squeeze(1)

        return combined_logits

    def inference(self, src, pos_embed, seq, vocab_embed, window, seq_format):
        """Enhanced inference with confidence-aware processing"""
        n, bs, c = src.shape
        memory = src
        confidence_list = []
        generated_tokens = []
        box_pos = [0, 1, 2, 3]
        center_pos = [0, 1] if seq_format != 'whxy' else [2, 3]

        # Determine sequence length based on confidence mode
        total_steps = self.num_coordinates + (1 if self.confidence_enabled else 0)

        for i in range(total_steps):
            # Prepare embeddings for current sequence
            if self.confidence_enabled and len(seq.shape) > 1:
                # Enhanced embedding preparation
                seq_data = self.prepare_input_sequence(seq[:, -4:] if seq.size(1) >= 4 else seq)
                tgt = self.embedding(
                    input_ids=seq_data['input_ids'][:, :seq.size(1)],
                    position_ids=seq_data['position_ids'][:, :seq.size(1)],
                    type_ids=seq_data['type_ids'][:, :seq.size(1)] if seq_data['type_ids'] is not None else None
                ).permute(1, 0, 2)
            else:
                # Legacy embedding
                tgt = self.embedding(seq).permute(1, 0, 2)

            query_embed = self.embedding.position_embeddings.weight.unsqueeze(1).repeat(1, bs, 1)
            tgt_mask = generate_square_subsequent_mask(len(tgt)).to(tgt.device)

            hs = self.body(tgt, memory, pos=pos_embed[:len(memory)],
                           query_pos=query_embed[:len(tgt)], tgt_mask=tgt_mask, memory_mask=None)

            # Get last position output
            last_output = hs.transpose(1, 2)[-1, :, -1, :]  # [batch, hidden_dim]

            # Shared processing
            shared_features = self.shared_head(last_output.unsqueeze(1))  # [batch, 1, hidden_dim]
            shared_features = shared_features.squeeze(1)  # [batch, hidden_dim]

            if i in box_pos:
                # Predict bbox token
                if self.confidence_enabled:
                    logits = self.bbox_head(shared_features)  # [batch, bbox_vocab_size]
                else:
                    # Use legacy vocab_embed for bbox
                    out = vocab_embed(last_output.unsqueeze(0).unsqueeze(2))
                    logits = out.softmax(-1)[:, :self.bins].squeeze(0).squeeze(1)

                # Apply window penalty if specified
                if i in center_pos and window is not None:
                    logits = logits * window

            elif i == self.num_coordinates and self.confidence_enabled:
                # Predict confidence token
                confidence_logits = self.confidence_head(shared_features)  # [batch, confidence_bins]
                confidence_logits = self.confidence_calibration(confidence_logits)

                # Apply confidence gating
                gate_weight = self.confidence_gate(shared_features)  # [batch, 1]
                logits = confidence_logits * gate_weight
            else:
                # Fallback to legacy approach
                out = vocab_embed(last_output.unsqueeze(0).unsqueeze(2))
                logits = out.softmax(-1).squeeze(0).squeeze(1)
                if i in box_pos:
                    logits = logits[:, :self.bins]
                elif self.confidence_enabled and i == self.num_coordinates:
                    logits = logits[:, self.confidence_vocab_offset:self.confidence_vocab_offset + self.confidence_bins]

            # Get top prediction
            if isinstance(logits, torch.Tensor) and len(logits.shape) > 1:
                confidence, token_generated = logits.topk(dim=-1, k=1)
            else:
                confidence, token_generated = logits.topk(dim=-1, k=1)

            seq = torch.cat([seq, token_generated], dim=-1)
            confidence_list.append(confidence)
            generated_tokens.append(token_generated)

        out_dict = {
            'pred_boxes': seq[:, -total_steps:] if total_steps <= self.num_coordinates else seq[:,
                                                                                            -self.num_coordinates:],
            'confidence': torch.cat(confidence_list, dim=-1),
            'tokens': torch.cat(generated_tokens, dim=-1)
        }
        return out_dict

    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor],
                     loss_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Compute confidence-aware loss"""
        losses = {}

        # Bbox loss
        if 'bbox_logits' in predictions and 'bbox_tokens' in targets:
            bbox_loss = F.cross_entropy(
                predictions['bbox_logits'].view(-1, self.bbox_vocab_size),
                targets['bbox_tokens'].view(-1),
                reduction='mean'
            )
            losses['bbox_loss'] = bbox_loss

        # Confidence loss
        if self.confidence_enabled and 'confidence_logits' in predictions and 'confidence_tokens' in targets:
            confidence_loss = F.cross_entropy(
                predictions['confidence_logits'].view(-1, self.confidence_bins),
                targets['confidence_tokens'].view(-1),
                reduction='mean'
            )
            losses['confidence_loss'] = confidence_loss

            # Confidence-weighted bbox loss (advanced)
            if 'bbox_logits' in predictions and 'bbox_loss' in losses:
                confidence_weights = F.softmax(predictions['confidence_logits'], dim=-1)
                max_confidence = torch.max(confidence_weights, dim=-1)[0]  # [batch, 1]
                weighted_bbox_loss = losses['bbox_loss'] * max_confidence.mean()
                losses['weighted_bbox_loss'] = weighted_bbox_loss

        # Combined loss
        total_loss = 0.0
        if 'bbox_loss' in losses:
            total_loss += loss_weights.get('bbox', 0.7) * losses['bbox_loss']

        if 'confidence_loss' in losses:
            total_loss += loss_weights.get('confidence', 0.3) * losses['confidence_loss']

        losses['total_loss'] = total_loss
        return losses

    def get_confidence_score(self, confidence_logits: torch.Tensor) -> torch.Tensor:
        """Convert confidence logits to confidence scores"""
        confidence_probs = F.softmax(confidence_logits, dim=-1)  # [batch, confidence_bins]

        # Weighted average of bin indices
        bin_indices = torch.arange(self.confidence_bins, device=confidence_logits.device).float()
        confidence_scores = torch.sum(confidence_probs * bin_indices, dim=-1)  # [batch]

        # Normalize to [0, 1]
        confidence_scores = confidence_scores / (self.confidence_bins - 1)
        return confidence_scores

    def adaptive_template_update_decision(self, confidence_scores: torch.Tensor,
                                          threshold: float = 0.7) -> torch.Tensor:
        """Make adaptive template update decisions based on confidence"""
        return confidence_scores > threshold


# Legacy compatibility wrapper
class SeqTrackDecoder(ConfidenceAwareSeqTrackDecoder):
    """Backward-compatible wrapper for the original SeqTrackDecoder"""

    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, activation="relu", normalize_before=False,
                 return_intermediate_dec=False, bins=1000, confidence_bins=100, num_frames=9):
        # Determine if confidence is enabled based on total bins
        confidence_enabled = (bins > 4000) or (confidence_bins > 0)

        super().__init__(
            d_model=d_model, nhead=nhead, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation,
            normalize_before=normalize_before, return_intermediate_dec=return_intermediate_dec,
            bins=bins - confidence_bins if confidence_enabled else bins,
            confidence_bins=confidence_bins, num_frames=num_frames,
            confidence_enabled=confidence_enabled
        )


def generate_square_subsequent_mask(sz):
    """Generate causal mask for autoregressive generation"""
    mask = torch.triu(torch.ones(sz, sz)) == 1
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output) if self.norm is not None else output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate[-1] = output

        return torch.stack(intermediate) if self.return_intermediate else output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos),
                                   self.with_pos_embed(memory, pos), memory,
                                   attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt

    def forward_pre(self, tgt, memory, tgt_mask=None, memory_mask=None,
                    tgt_key_padding_mask=None, memory_key_padding_mask=None,
                    pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt2, query_pos),
                                   self.with_pos_embed(memory, pos), memory,
                                   attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def build_decoder(cfg):
    return SeqTrackDecoder(
        d_model=cfg.MODEL.HIDDEN_DIM,
        dropout=cfg.MODEL.DECODER.DROPOUT,
        nhead=cfg.MODEL.DECODER.NHEADS,
        dim_feedforward=cfg.MODEL.DECODER.DIM_FEEDFORWARD,
        num_decoder_layers=cfg.MODEL.DECODER.DEC_LAYERS,
        normalize_before=cfg.MODEL.DECODER.PRE_NORM,
        return_intermediate_dec=False,
        bins=cfg.MODEL.BINS,
        confidence_bins=getattr(cfg.MODEL, 'CONFIDENCE_BINS', 100),
        num_frames=cfg.DATA.SEARCH.NUMBER
    )


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# Utility functions for confidence processing
def tokenize_confidence(confidence_scores: torch.Tensor, num_bins: int = 10) -> torch.Tensor:
    """Tokenize continuous confidence scores into discrete bins"""
    confidence_scores = torch.clamp(confidence_scores, 0.0, 1.0)
    confidence_tokens = (confidence_scores * (num_bins - 1)).long()
    return confidence_tokens


def detokenize_confidence(confidence_tokens: torch.Tensor, num_bins: int = 10) -> torch.Tensor:
    """Convert discrete confidence tokens back to continuous scores"""
    confidence_scores = confidence_tokens.float() / (num_bins - 1)
    return confidence_scores
