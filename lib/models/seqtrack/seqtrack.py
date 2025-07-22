import torch
import math
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from lib.utils.misc import NestedTensor
from lib.models.seqtrack.encoder import build_encoder
from .decoder import build_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed


class SEQTRACK(nn.Module):
    """ Enhanced SeqTrack with Confidence-Aware Sequence Modeling """

    def __init__(self, encoder, decoder, hidden_dim,
                 bins=1000, confidence_bins=100, feature_type='x', num_frames=1, num_template=1):
        super().__init__()

        # Encoder setup
        self.encoder = encoder
        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.side_fx = int(math.sqrt(self.num_patch_x))
        self.side_fz = int(math.sqrt(self.num_patch_z))
        self.hidden_dim = hidden_dim

        # Vocabulary and confidence setup
        self.bbox_bins = bins
        self.confidence_bins = confidence_bins
        self.confidence_enabled = hasattr(decoder, 'confidence_enabled') and decoder.confidence_enabled

        if self.confidence_enabled:
            self.total_bins = self.bbox_bins + self.confidence_bins + 2  # +2 for start/end
            self.confidence_vocab_offset = self.bbox_bins
        else:
            self.total_bins = self.bbox_bins + 2  # +2 for start/end
            self.confidence_vocab_offset = None

        # Bottleneck and vocabulary embedding
        self.bottleneck = nn.Linear(encoder.num_channels, hidden_dim)

        # Enhanced vocabulary embedding with dual-head support
        if self.confidence_enabled:
            # Separate vocab embeddings for bbox and confidence
            self.bbox_vocab_embed = MLP(hidden_dim, hidden_dim, self.bbox_bins, 3)
            self.confidence_vocab_embed = MLP(hidden_dim, hidden_dim, self.confidence_bins, 3)
            # Combined vocab embed for legacy compatibility
            self.vocab_embed = MLP(hidden_dim, hidden_dim, self.total_bins, 3)
        else:
            self.vocab_embed = MLP(hidden_dim, hidden_dim, self.total_bins, 3)

        self.decoder = decoder
        self.num_frames = num_frames
        self.num_template = num_template
        self.feature_type = feature_type

        # Positional embeddings setup
        if self.feature_type == 'x':
            num_patches = self.num_patch_x * self.num_frames
        elif self.feature_type == 'xz':
            num_patches = self.num_patch_x * self.num_frames + self.num_patch_z * self.num_template
        elif self.feature_type == 'token':
            num_patches = 1
        else:
            raise ValueError('illegal feature type')

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        pos_embed = get_sinusoid_encoding_table(num_patches, self.pos_embed.shape[-1], cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, images_list=None, xz=None, seq=None, mode="encoder", **kwargs):
        if mode == "encoder":
            return self.forward_encoder(images_list)
        elif mode == "decoder":
            return self.forward_decoder(xz, seq, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward_encoder(self, images_list):
        """Forward pass through encoder"""
        xz = self.encoder(images_list)
        return xz

    def forward_decoder(self, xz, sequence, bbox_tokens=None, confidence_tokens=None, is_training=True):
        """Enhanced decoder forward with confidence-aware processing"""
        xz_mem = xz[-1]
        B, _, _ = xz_mem.shape

        # Extract relevant features based on feature type
        if self.feature_type == 'x':
            dec_mem = xz_mem[:, 0:self.num_patch_x * self.num_frames]
        elif self.feature_type == 'xz':
            dec_mem = xz_mem
        elif self.feature_type == 'token':
            dec_mem = xz_mem.mean(1).unsqueeze(1)
        else:
            raise ValueError('illegal feature type')

        # Apply bottleneck if necessary
        if dec_mem.shape[-1] != self.hidden_dim:
            dec_mem = self.bottleneck(dec_mem)
        dec_mem = dec_mem.permute(1, 0, 2)

        # Enhanced decoder forward with confidence support
        if self.confidence_enabled and hasattr(self.decoder, 'forward_training') and is_training:
            # Use confidence-aware training mode
            out = self.decoder.forward(
                src=dec_mem,
                pos_embed=self.pos_embed.permute(1, 0, 2).expand(-1, B, -1),
                seq=sequence,
                bbox_tokens=bbox_tokens,
                confidence_tokens=confidence_tokens,
                is_training=True
            )

            # Apply separate vocab embeddings
            if isinstance(out, dict):
                if 'bbox_logits' in out:
                    out['bbox_logits'] = out['bbox_logits']  # Already processed by decoder heads
                if 'confidence_logits' in out:
                    out['confidence_logits'] = out['confidence_logits']  # Already processed by decoder heads
                return out
            else:
                # Fallback to combined vocab embedding
                return self.vocab_embed(out)
        else:
            # Legacy mode or inference mode
            out = self.decoder(dec_mem, self.pos_embed.permute(1, 0, 2).expand(-1, B, -1), sequence)
            return self.vocab_embed(out)

    def inference_decoder(self, xz, sequence, window=None, seq_format='xywh'):
        """Enhanced inference with confidence-aware processing"""
        xz_mem = xz[-1]
        B, _, _ = xz_mem.shape

        # Extract relevant features based on feature type
        if self.feature_type == 'x':
            dec_mem = xz_mem[:, 0:self.num_patch_x]
        elif self.feature_type == 'xz':
            dec_mem = xz_mem
        elif self.feature_type == 'token':
            dec_mem = xz_mem.mean(1).unsqueeze(1)
        else:
            raise ValueError('illegal feature type')

        # Apply bottleneck if necessary
        if dec_mem.shape[-1] != self.hidden_dim:
            dec_mem = self.bottleneck(dec_mem)
        dec_mem = dec_mem.permute(1, 0, 2)

        # Enhanced inference with confidence support
        if self.confidence_enabled and hasattr(self.decoder, 'inference'):
            # Use confidence-aware inference
            out = self.decoder.inference(
                dec_mem,
                self.pos_embed.permute(1, 0, 2).expand(-1, B, -1),
                sequence,
                self.vocab_embed,
                window,
                seq_format
            )
        else:
            # Legacy inference
            out = self.decoder.inference(
                dec_mem,
                self.pos_embed.permute(1, 0, 2).expand(-1, B, -1),
                sequence,
                self.vocab_embed,
                window,
                seq_format
            )

        return out

    def compute_confidence_metrics(self, predictions: Dict[str, torch.Tensor],
                                   targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute confidence-specific evaluation metrics"""
        metrics = {}

        if not self.confidence_enabled:
            return metrics

        if 'confidence_logits' in predictions and 'confidence_tokens' in targets:
            # Confidence accuracy
            conf_preds = torch.argmax(predictions['confidence_logits'], dim=-1)
            conf_targets = targets['confidence_tokens']
            conf_accuracy = (conf_preds == conf_targets).float().mean()
            metrics['confidence_accuracy'] = conf_accuracy.item()

            # Confidence distribution entropy
            conf_probs = F.softmax(predictions['confidence_logits'], dim=-1)
            conf_entropy = -torch.sum(conf_probs * torch.log(conf_probs + 1e-8), dim=-1).mean()
            metrics['confidence_entropy'] = conf_entropy.item()

            # Mean confidence score
            conf_scores = self.decoder.get_confidence_score(predictions['confidence_logits'])
            metrics['mean_confidence'] = conf_scores.mean().item()

        return metrics

    def adaptive_template_update(self, confidence_scores: torch.Tensor,
                                 threshold: float = 0.7) -> torch.Tensor:
        """Make adaptive template update decisions based on confidence"""
        if not self.confidence_enabled:
            return torch.ones_like(confidence_scores, dtype=torch.bool)

        return self.decoder.adaptive_template_update_decision(confidence_scores, threshold)


class MLP(nn.Module):
    """ Multi-layer perceptron """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_seqtrack(cfg):
    """Build SeqTrack model with confidence awareness"""
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg)

    # Extract confidence parameters from config
    confidence_bins = getattr(cfg.MODEL, "CONFIDENCE_BINS", 100)
    if hasattr(cfg.MODEL, "CONFIDENCE") and hasattr(cfg.MODEL.CONFIDENCE, "BINS"):
        confidence_bins = cfg.MODEL.CONFIDENCE.BINS

    model = SEQTRACK(
        encoder,
        decoder,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        bins=cfg.MODEL.BINS,
        confidence_bins=confidence_bins,
        feature_type=cfg.MODEL.FEATURE_TYPE,
        num_frames=cfg.DATA.SEARCH.NUMBER,
        num_template=cfg.DATA.TEMPLATE.NUMBER
    )
    return model
