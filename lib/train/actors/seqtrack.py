from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
import torch.nn.functional as F

class SeqTrackActor(BaseActor):
    """ Actor for training the SeqTrack with confidence awareness """
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.BINS = cfg.MODEL.BINS
        self.CONF_BINS = cfg.MODEL.CONFIDENCE.BINS
        self.seq_format = cfg.DATA.SEQ_FORMAT
        self.conf_token_start = self.BINS - self.CONF_BINS
        self.conf_loss_weight = cfg.TRAIN.CONFIDENCE.WEIGHT
        self.ce_weight = cfg.TRAIN.CE_WEIGHT

    def __call__(self, data):
        outputs, target_seqs = self.forward_pass(data)
        loss, status = self.compute_losses(outputs, target_seqs)
        return loss, status

    def forward_pass(self, data):
        n, b, _, _, _ = data['search_images'].shape
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])
        search_list = search_img.split(b, dim=0)
        template_img = data['template_images'].view(-1, *data['template_images'].shape[2:])
        template_list = template_img.split(b, dim=0)

        feature_xz = self.net(images_list=template_list + search_list, mode='encoder')

        targets = data['search_anno'].permute(1, 0, 2).reshape(-1, data['search_anno'].shape[2])
        targets = box_xywh_to_xyxy(targets)
        targets = torch.clamp(targets, min=0.0, max=1.0)

        if self.seq_format != 'corner':
            targets = box_xyxy_to_cxcywh(targets)

        box = (targets[:, :4] * (self.BINS - 1)).int()
        conf = targets[:, 4].int() + self.conf_token_start

        if self.seq_format == 'whxy':
            box = box[:, [2, 3, 0, 1]]

        batch = box.shape[0]
        start = torch.full((batch, 1), self.BINS + 1, dtype=torch.int64, device=box.device)
        end = torch.full((batch, 1), self.BINS, dtype=torch.int64, device=box.device)

        input_seqs = torch.cat([start, box], dim=1)
        input_seqs = input_seqs.view(b, n, -1).flatten(1)

        target_seqs = torch.cat([box, conf.unsqueeze(1), end], dim=1)
        target_seqs = target_seqs.view(b, n, -1).flatten().long()

        outputs = self.net(xz=feature_xz, seq=input_seqs, mode="decoder")
        outputs = outputs[-1].reshape(-1, self.BINS + self.CONF_BINS + 2)

        return outputs, target_seqs

    def compute_losses(self, outputs, targets_seq, return_status=True):
        pad_mask = targets_seq != 0
        outputs = outputs[pad_mask]
        targets_seq = targets_seq[pad_mask]

        is_conf = targets_seq >= self.conf_token_start

        ce_loss = F.cross_entropy(outputs, targets_seq, reduction='none')
        main_loss = ce_loss[~is_conf].mean()
        conf_loss = ce_loss[is_conf].mean() if is_conf.any() else torch.tensor(0.0, device=main_loss.device)

        total_loss = self.ce_weight * main_loss + self.conf_loss_weight * conf_loss

        outputs_softmax = outputs.softmax(-1)[:, :self.BINS]
        preds = outputs_softmax.argmax(-1).reshape(-1, 6)[:, :4]
        targets = targets_seq.reshape(-1, 6)[:, :4]
        boxes_pred = box_cxcywh_to_xyxy(preds)
        boxes_target = box_cxcywh_to_xyxy(targets)
        iou = box_iou(boxes_pred, boxes_target)[0].mean()

        if return_status:
            status = {"Loss/total": total_loss.item(),
                      "IoU": iou.item(),
                      "Loss/main": main_loss.item(),
                      "Loss/conf": conf_loss.item()}
            return total_loss, status
        else:
            return total_loss

    def to(self, device):
        self.net.to(device)
        self.objective['ce'].to(device)
