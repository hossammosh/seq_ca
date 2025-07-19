import os
class EnvironmentSettings:
    def __init__(self):

        self.workspace_dir = os.getcwd()  # This will be '/mnt/e/current_research/reducedDS'
        self.tensorboard_dir = os.path.join(self.workspace_dir, 'tensorboard')
        self.pretrained_networks = os.path.join(self.workspace_dir, 'pretrained_networks')
        #checkpoints
        self.lasot_dir = os.path.join(os.getcwd(), 'data', 'lasot')

        #self.lasot_dir = '/mnt/e/current_research/seq/data/lasot'
        # self.got10k_dir = '/mnt/e/current_research/seq/data/got10k'
        #self.lasot_lmdb_dir = '/mnt/e/current_research/seq/data/lasot_lmdb'
        # self.got10k_lmdb_dir = '/mnt/e/current_research/seq/data/got10k_lmdb'
        # self.trackingnet_dir = '/mnt/e/current_research/seq/data/trackingnet'
        # self.trackingnet_lmdb_dir = '/mnt/e/current_research/seq/data/trackingnet_lmdb'
        # self.coco_dir = '/mnt/e/current_research/seq/data/coco'
        # self.coco_lmdb_dir = '/mnt/e/current_research/seq/data/coco_lmdb'
        # self.imagenet1k_dir = '/mnt/e/current_research/seq/data/imagenet1k'
        # self.imagenet22k_dir = '/mnt/e/current_research/seq/data/imagenet22k'
        # self.lvis_dir = ''
        # self.sbd_dir = ''
        # self.imagenet_dir = '/mnt/e/current_research/seq/data/vid'
        # self.imagenet_lmdb_dir = '/mnt/e/current_research/seq/data/vid_lmdb'
        # self.imagenetdet_dir = ''
        # self.ecssd_dir = ''
        # self.hkuis_dir = ''
        # self.msra10k_dir = ''
        # self.davis_dir = ''
        # self.youtubevos_dir = ''