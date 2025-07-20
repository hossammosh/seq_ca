from easydict import EasyDict as edict
import yaml

cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.HIDDEN_DIM = 256 # hidden dimension for the decoder and vocabulary
cfg.MODEL.BINS = 4010 # number of discrete bins
cfg.MODEL.FEATURE_TYPE = "x" # the input feature to decoder. x, xz, or token

# MODEL.CONFIDENCE - Add confidence-related parameters
cfg.MODEL.CONFIDENCE = edict()
cfg.MODEL.CONFIDENCE.ENABLED = True  # Enable confidence-aware modeling
cfg.MODEL.CONFIDENCE.BINS = 10  # Number of discrete confidence levels (0-9)
cfg.MODEL.CONFIDENCE.TOKEN_TYPE = "append"  # How to add confidence: "append" or "embed"

# MODEL.ENCODER
# for more customization for encoder, please modify lib/models/seqtrack/vit.py
cfg.MODEL.ENCODER = edict()
cfg.MODEL.ENCODER.TYPE = "vit_base_patch16" # encoder model
cfg.MODEL.ENCODER.DROP_PATH = 0
cfg.MODEL.ENCODER.PRETRAIN_TYPE = "mae" # mae, default, or scratch
cfg.MODEL.ENCODER.STRIDE = 16
cfg.MODEL.ENCODER.USE_CHECKPOINT = False # to save the memory.

# MODEL.DECODER
cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.NHEADS = 8
cfg.MODEL.DECODER.DROPOUT = 0.1
cfg.MODEL.DECODER.DIM_FEEDFORWARD = 1024
cfg.MODEL.DECODER.DEC_LAYERS = 6
cfg.MODEL.DECODER.PRE_NORM = False
# Add confidence head parameters
cfg.MODEL.DECODER.CONFIDENCE_HEAD = edict()
cfg.MODEL.DECODER.CONFIDENCE_HEAD.ENABLED = True
cfg.MODEL.DECODER.CONFIDENCE_HEAD.HIDDEN_DIM = 256
cfg.MODEL.DECODER.CONFIDENCE_HEAD.NUM_LAYERS = 2

# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.ENCODER_MULTIPLIER = 0.1  # encoder's LR = this factor * LR
cfg.TRAIN.FREEZE_ENCODER = False # for freezing the parameters of encoder
cfg.TRAIN.ENCODER_OPEN = [] # only for debug, open some layers of encoder when FREEZE_ENCODER is True
cfg.TRAIN.CE_WEIGHT = 0.7  # Reduced from 1.0 to accommodate confidence loss weight
cfg.TRAIN.PRINT_INTERVAL = 50 # interval to print the training log
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
# Add confidence training parameters
cfg.TRAIN.CONFIDENCE = edict()
cfg.TRAIN.CONFIDENCE.WEIGHT = 0.3  # Weight for confidence loss component
cfg.TRAIN.CONFIDENCE.LOSS_TYPE = "cross_entropy"  # Loss function for confidence
cfg.TRAIN.CONFIDENCE.FOCAL_ALPHA = 0.25  # For focal loss (if used)
cfg.TRAIN.CONFIDENCE.FOCAL_GAMMA = 2.0   # For focal loss (if used)

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
cfg.DATA.SAMPLER_MODE = "order"
cfg.DATA.LOADER = "tracking"
cfg.DATA.SEQ_FORMAT = "xywhc"  # Updated from "xywh" to include confidence
cfg.DATA.SEQUENCE_LENGTH = 5  # [x, y, w, h, confidence]
# Add confidence-related data processing
cfg.DATA.CONFIDENCE = edict()
cfg.DATA.CONFIDENCE.IOU_THRESHOLD = 0.7  # IoU threshold for high confidence
cfg.DATA.CONFIDENCE.SIZE_CONSISTENCY_WEIGHT = 0.3  # Weight for size consistency
cfg.DATA.CONFIDENCE.TEMPORAL_WEIGHT = 0.2  # Weight for temporal consistency

# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000

# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.NUMBER = 1  #number of search region, only support 1 for now.
cfg.DATA.SEARCH.SIZE = 256
cfg.DATA.SEARCH.FACTOR = 4.0
cfg.DATA.SEARCH.CENTER_JITTER = 3.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5

# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 256
cfg.DATA.TEMPLATE.FACTOR = 4.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 4.0
cfg.TEST.TEMPLATE_SIZE = 256
cfg.TEST.SEARCH_FACTOR = 4.0
cfg.TEST.SEARCH_SIZE = 256
cfg.TEST.EPOCH = 500
cfg.TEST.WINDOW = False # window penalty
cfg.TEST.NUM_TEMPLATES = 1
# Add confidence-aware inference parameters
cfg.TEST.CONFIDENCE = edict()
cfg.TEST.CONFIDENCE.ENABLED = True
cfg.TEST.CONFIDENCE.THRESHOLD = 0.7  # Confidence threshold for template update
cfg.TEST.CONFIDENCE.ADAPTIVE_UPDATE = True  # Enable adaptive template update
cfg.TEST.CONFIDENCE.MIN_CONFIDENCE = 0.3  # Minimum confidence for valid tracking
# Add confidence-specific evaluation metrics
cfg.TEST.CONFIDENCE.METRICS = edict()
cfg.TEST.CONFIDENCE.METRICS.COMPUTE_PRECISION = True
cfg.TEST.CONFIDENCE.METRICS.COMPUTE_RECALL = True
cfg.TEST.CONFIDENCE.METRICS.COMPUTE_F1 = True
cfg.TEST.CONFIDENCE.METRICS.CONFIDENCE_HISTOGRAM = True

# TEST.UPDATE_INTERVALS
cfg.TEST.UPDATE_INTERVALS = edict()
cfg.TEST.UPDATE_INTERVALS.DEFAULT = 9999
cfg.TEST.UPDATE_INTERVALS.LASOT = 450
cfg.TEST.UPDATE_INTERVALS.LASOT_EXTENSION_SUBSET = 9999
cfg.TEST.UPDATE_INTERVALS.GOT10K_TEST = 1
cfg.TEST.UPDATE_INTERVALS.TRACKINGNET = 25
cfg.TEST.UPDATE_INTERVALS.TNL2K = 25
cfg.TEST.UPDATE_INTERVALS.OTB = 9999
cfg.TEST.UPDATE_INTERVALS.NFS = 9999
cfg.TEST.UPDATE_INTERVALS.UAV = 9999
cfg.TEST.UPDATE_INTERVALS.VOT20 = 10
cfg.TEST.UPDATE_INTERVALS.VOT21 = 10
cfg.TEST.UPDATE_INTERVALS.VOT22 = 10

# TEST.UPDATE_THRESHOLD - Original update thresholds
cfg.TEST.UPDATE_THRESHOLD = edict()
cfg.TEST.UPDATE_THRESHOLD.DEFAULT = 0.6
cfg.TEST.UPDATE_THRESHOLD.VOT20 = 0.475
cfg.TEST.UPDATE_THRESHOLD.VOT21 = 0.475
cfg.TEST.UPDATE_THRESHOLD.VOT22 = 0.475

# TEST.UPDATE_THRESHOLD_CONFIDENCE - Confidence thresholds for template update
cfg.TEST.UPDATE_THRESHOLD_CONFIDENCE = edict()
cfg.TEST.UPDATE_THRESHOLD_CONFIDENCE.DEFAULT = 0.7
cfg.TEST.UPDATE_THRESHOLD_CONFIDENCE.VOT20 = 0.6
cfg.TEST.UPDATE_THRESHOLD_CONFIDENCE.VOT21 = 0.6
cfg.TEST.UPDATE_THRESHOLD_CONFIDENCE.VOT22 = 0.6

# Add backward compatibility flags
cfg.MODEL.LEGACY_MODE = False  # Set to True to use original SeqTrack
cfg.MODEL.CONFIDENCE_BACKWARD_COMPATIBLE = True  # Gradual transition support

# Add debugging parameters
cfg.DEBUG = edict()
cfg.DEBUG.CONFIDENCE = edict()
cfg.DEBUG.CONFIDENCE.VISUALIZE_PREDICTIONS = False
cfg.DEBUG.CONFIDENCE.SAVE_CONFIDENCE_MAPS = False
cfg.DEBUG.CONFIDENCE.LOG_CONFIDENCE_DISTRIBUTION = True

def get_confidence_vocab_offset():
    """Get the starting index for confidence tokens in vocabulary"""
    return cfg.MODEL.BINS - cfg.MODEL.CONFIDENCE.BINS

def get_total_sequence_length():
    """Get total sequence length including confidence"""
    if cfg.MODEL.CONFIDENCE.ENABLED:
        return 5  # [start, x, y, w, h, confidence, end] - start/end not counted in seq
    else:
        return 4  # [start, x, y, w, h, end] - start/end not counted in seq

def get_confidence_loss_weight():
    """Get the weight for confidence loss"""
    return cfg.TRAIN.CONFIDENCE.WEIGHT if cfg.MODEL.CONFIDENCE.ENABLED else 0.0

def validate_confidence_config():
    """Validate confidence configuration parameters"""
    assert cfg.MODEL.CONFIDENCE.BINS > 0, "Confidence bins must be positive"
    assert cfg.MODEL.BINS >= 4000 + cfg.MODEL.CONFIDENCE.BINS, "Total bins too small"
    assert cfg.TRAIN.CE_WEIGHT + cfg.TRAIN.CONFIDENCE.WEIGHT <= 1.0, "Loss weights sum > 1.0"
    assert 0.0 <= cfg.TEST.CONFIDENCE.THRESHOLD <= 1.0, "Invalid confidence threshold"

def update_legacy_parameters():
    """Update parameters for backward compatibility"""
    if not cfg.MODEL.CONFIDENCE.ENABLED:
        cfg.MODEL.BINS = 4000  # Reset to original
        cfg.DATA.SEQ_FORMAT = "xywh"  # Reset to original
        cfg.DATA.SEQUENCE_LENGTH = 4  # Reset to original

def initialize_confidence_config():
    """Initialize confidence-aware configuration"""
    # Validate configuration
    validate_confidence_config()
    
    # Update legacy parameters if needed
    if cfg.MODEL.LEGACY_MODE:
        update_legacy_parameters()
    
    # Print configuration summary
    if cfg.MODEL.CONFIDENCE.ENABLED:
        print(f"Confidence-Aware SeqTrack Configuration:")
        print(f"  - Confidence bins: {cfg.MODEL.CONFIDENCE.BINS}")
        print(f"  - Total vocabulary: {cfg.MODEL.BINS}")
        print(f"  - Confidence loss weight: {cfg.TRAIN.CONFIDENCE.WEIGHT}")
        print(f"  - Sequence format: {cfg.DATA.SEQ_FORMAT}")
        print(f"  - Adaptive update threshold: {cfg.TEST.CONFIDENCE.THRESHOLD}")

def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return

def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)

def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return

def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)
