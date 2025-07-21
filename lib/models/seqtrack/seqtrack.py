import torch
import math
from torch import nn
import torch.nn.functional as F

from lib.utils.misc import NestedTensor
from lib.models.seqtrack.encoder import build_encoder
from .decoder import build_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed


class SEQTRACK(nn.Module):
    """ This is the base class for SeqTrack """
    def __init__(self, encoder, decoder, hidden_dim,
                 bins=1000, confidence_bins=100, feature_type='x', num_frames=1, num_template=1):
        super().__init__()
        self.encoder = encoder
        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.side_fx = int(math.sqrt(self.num_patch_x))
        self.side_fz = int(math.sqrt(self.num_patch_z))
        self.hidden_dim = hidden_dim
        self.bbox_bins = bins
        self.confidence_bins = confidence_bins
        self.total_bins = self.bbox_bins + self.confidence_bins + 2  # +2 for [start] and [end] tokens
        self.conf_token_start = self.bbox_bins + 1
        self.bottleneck = nn.Linear(encoder.num_channels, hidden_dim)
        self.vocab_embed = MLP(hidden_dim, hidden_dim, self.total_bins, 3)

        self.decoder = decoder
        self.num_frames = num_frames
        self.num_template = num_template
        self.feature_type = feature_type

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

    def forward(self, images_list=None, xz=None, seq=None, mode="encoder"):
        if mode == "encoder":
            return self.forward_encoder(images_list)
        elif mode == "decoder":
            return self.forward_decoder(xz, seq)
        else:
            raise ValueError

    def forward_encoder(self, images_list):
        xz = self.encoder(images_list)
        return xz

    def forward_decoder(self, xz, sequence):
        xz_mem = xz[-1]
        B, _, _ = xz_mem.shape

        if self.feature_type == 'x':
            dec_mem = xz_mem[:,0:self.num_patch_x * self.num_frames]
        elif self.feature_type == 'xz':
            dec_mem = xz_mem
        elif self.feature_type == 'token':
            dec_mem = xz_mem.mean(1).unsqueeze(1)
        else:
            raise ValueError('illegal feature type')

        if dec_mem.shape[-1] != self.hidden_dim:
            dec_mem = self.bottleneck(dec_mem)
        dec_mem = dec_mem.permute(1,0,2)

        out = self.decoder(dec_mem, self.pos_embed.permute(1,0,2).expand(-1,B,-1), sequence)
        out = self.vocab_embed(out)

        return out

    def inference_decoder(self, xz, sequence, window=None, seq_format='xywh'):
        xz_mem = xz[-1]
        B, _, _ = xz_mem.shape

        if self.feature_type == 'x':
            dec_mem = xz_mem[:,0:self.num_patch_x]
        elif self.feature_type == 'xz':
            dec_mem = xz_mem
        elif self.feature_type == 'token':
            dec_mem = xz_mem.mean(1).unsqueeze(1)
        else:
            raise ValueError('illegal feature type')

        if dec_mem.shape[-1] != self.hidden_dim:
            dec_mem = self.bottleneck(dec_mem)
        dec_mem = dec_mem.permute(1,0,2)

        out = self.decoder.inference(dec_mem,
                                    self.pos_embed.permute(1,0,2).expand(-1,B,-1),
                                    sequence, self.vocab_embed,
                                    window, seq_format)

        return out


class MLP(nn.Module):
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
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg)
    model = SEQTRACK(
        encoder,
        decoder,
        hidden_dim=cfg.MODEL.HIDDEN_DIM,
        bins=cfg.MODEL.BINS,
        confidence_bins=getattr(cfg.MODEL, "CONFIDENCE_BINS", 100),
        feature_type=cfg.MODEL.FEATURE_TYPE,
        num_frames=cfg.DATA.SEARCH.NUMBER,
        num_template=cfg.DATA.TEMPLATE.NUMBER
    )
    return model
