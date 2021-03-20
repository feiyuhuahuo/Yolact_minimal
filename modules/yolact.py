import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import mask_proto_net, extra_head_net
from modules.backbone import construct_backbone
from utils.box_utils import match, crop, make_anchors
import pdb


class Concat(nn.Module):
    def __init__(self, nets, extra_params):
        super().__init__()

        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params

    def forward(self, x):
        return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)


class InterpolateModule(nn.Module):  # A module version of F.interpolate.
    def __init__(self, *args, **kwdargs):
        super().__init__()
        self.args = args
        self.kwdargs = kwdargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwdargs)


def make_net(in_channels, cfg_net, include_last_relu=True):
    def make_layer(layer_cfg):
        nonlocal in_channels

        if isinstance(layer_cfg[0], str):
            layer_name = layer_cfg[0]

            if layer_name == 'cat':
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                layer = Concat([net[0] for net in nets], layer_cfg[2])
                num_channels = sum([net[1] for net in nets])
        else:
            num_channels = layer_cfg[0]
            kernel_size = layer_cfg[1]

            if kernel_size > 0:
                layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])

            else:
                if num_channels is None:
                    layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False,
                                              **layer_cfg[2])
                else:
                    layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])

        in_channels = num_channels if num_channels is not None else in_channels

        return [layer, nn.ReLU(inplace=True)]

    # Use sum to concat all the component layer lists
    net = sum([make_layer(x) for x in cfg_net], [])  # x: (256, 3, {'padding': 1})

    if not include_last_relu:
        net = net[:-1]

    return nn.Sequential(*net), in_channels


class PredictionModule(nn.Module):
    def __init__(self, cfg, in_channels=256, coef_dim=32):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.coef_dim = coef_dim

        self.upfeature, out_channels = make_net(in_channels, extra_head_net)
        self.bbox_layer = nn.Conv2d(out_channels, len(cfg.aspect_ratios) * 4, kernel_size=3, padding=1)
        self.conf_layer = nn.Conv2d(out_channels, len(cfg.aspect_ratios) * self.num_classes, kernel_size=3, padding=1)
        self.mask_layer = nn.Conv2d(out_channels, len(cfg.aspect_ratios) * self.coef_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upfeature(x)
        conf = self.conf_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_classes)
        box = self.bbox_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
        coef = self.mask_layer(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.coef_dim)
        coef = torch.tanh(coef)
        return conf, box, coef


class FPN(nn.Module):
    """
    The FPN here is slightly different from the FPN introduced in https://arxiv.org/pdf/1612.03144.pdf.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.num_downsample = 2
        self.in_channels = in_channels

        self.lat_layers = nn.ModuleList([nn.Conv2d(x, 256, kernel_size=1) for x in reversed(self.in_channels)])
        self.pred_layers = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=3, padding=1) for _ in self.in_channels])
        self.downsample_layers = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)
                                                for _ in range(self.num_downsample)])

    def forward(self, backbone_outs):
        out = []
        x = torch.zeros(1, device=backbone_outs[0].device)
        for i in range(len(backbone_outs)):
            out.append(x)

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(backbone_outs)  # convouts: C3, C4, C5

        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(backbone_outs) - 1:
                _, _, h, w = backbone_outs[j].size()
                x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

            x = x + lat_layer(backbone_outs[j])

            out[j] = x

        j = len(backbone_outs)
        for pred_layer in self.pred_layers:
            j -= 1
            out[j] = F.relu(pred_layer(out[j]))

        for layer in self.downsample_layers:
            out.append(layer(out[-1]))

        return out


class Yolact(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.anchors = []
        self.backbone = construct_backbone(cfg.__class__.__name__, (1, 2, 3))
        self.proto_net, coef_dim = make_net(256, mask_proto_net, include_last_relu=False)
        self.fpn = FPN([512, 1024, 2048])
        self.prediction_layers = nn.ModuleList()
        self.prediction_layers.append(PredictionModule(cfg, coef_dim=coef_dim))

        if cfg.mode == 'train':
            ch_out = cfg.num_classes - 1
            self.semantic_seg_conv = nn.Conv2d(256, ch_out, kernel_size=1)

    def load_weights(self, path, cuda):
        if cuda:
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location='cpu')

        for key in list(state_dict.keys()):
            # 'fpn.downsample_layers.2.weight' and 'fpn.downsample_layers.2.bias'
            # in the pretrained .pth are redundant, remove them
            if key.startswith('fpn.downsample_layers.'):
                if int(key.split('.')[2]) >= 2:
                    del state_dict[key]

            if self.cfg.mode != 'train' and key.startswith('semantic_seg_conv'):
                del state_dict[key]

        self.load_state_dict(state_dict)

    def init_weights(self, backbone_path):
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path)
        # Initialize the rest conv layers with xavier
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, img, box_classes=None, masks_gt=None):
        outs = self.backbone(img)
        outs = self.fpn(outs[1:4])
        if isinstance(self.anchors, list):
            for i, shape in enumerate([list(aa.shape) for aa in outs]):
                self.anchors += make_anchors(self.cfg, shape[2], shape[3], self.cfg.scales[i])

            self.anchors = torch.tensor(self.anchors, device=outs[0].device).reshape(-1, 4)

        # outs[0]: [2, 256, 69, 69], the feature map from P3
        proto_out = self.proto_net(outs[0])  # proto_out: (n, 32, 138, 138)
        proto_out = F.relu(proto_out, inplace=True)
        proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

        class_pred, box_pred, coef_pred = [], [], []

        for aa in outs:
            class_p, box_p, coef_p = self.prediction_layers[0](aa)
            class_pred.append(class_p)
            box_pred.append(box_p)
            coef_pred.append(coef_p)

        class_pred = torch.cat(class_pred, dim=1)
        box_pred = torch.cat(box_pred, dim=1)
        coef_pred = torch.cat(coef_pred, dim=1)

        if self.training:
            seg_pred = self.semantic_seg_conv(outs[0])
            return self.compute_loss(class_pred, box_pred, coef_pred, proto_out, seg_pred, box_classes, masks_gt)
        else:
            class_pred = F.softmax(class_pred, -1)
            return class_pred, box_pred, coef_pred, proto_out, self.anchors

    def compute_loss(self, class_p, box_p, coef_p, proto_p, seg_p, box_class, mask_gt):
        device = class_p.device
        class_gt = [None] * len(box_class)
        batch_size = box_p.size(0)
        num_anchors = self.anchors.shape[0]

        all_offsets = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)
        conf_gt = torch.zeros((batch_size, num_anchors), dtype=torch.int64, device=device)
        anchor_max_gt = torch.zeros((batch_size, num_anchors, 4), dtype=torch.float32, device=device)
        anchor_max_i = torch.zeros((batch_size, num_anchors), dtype=torch.int64, device=device)

        for i in range(batch_size):
            box_gt = box_class[i][:, :-1]
            class_gt[i] = box_class[i][:, -1].long()

            all_offsets[i], conf_gt[i], anchor_max_gt[i], anchor_max_i[i] = match(self.cfg, box_gt,
                                                                                  self.anchors, class_gt[i])

        # all_offsets: the transformed box coordinate offsets of each pair of anchor and gt box
        # conf_gt: the foreground and background labels according to the 'pos_thre' and 'neg_thre',
        #          '0' means background, '>0' means foreground.
        # anchor_max_gt: the corresponding max IoU gt box for each anchor
        # anchor_max_i: the index of the corresponding max IoU gt box for each anchor
        assert (not all_offsets.requires_grad) and (not conf_gt.requires_grad) and (not anchor_max_i.requires_grad), \
            'Incorrect computation graph, check the grad.'

        # only compute losses from positive samples
        pos_bool = conf_gt > 0  # (n, 19248)

        loss_c = self.category_loss(class_p, conf_gt, pos_bool)
        loss_b = self.box_loss(box_p, all_offsets, pos_bool)
        loss_m = self.lincomb_mask_loss(pos_bool, anchor_max_i, coef_p, proto_p, mask_gt, anchor_max_gt)
        loss_s = self.semantic_seg_loss(seg_p, mask_gt, class_gt)

        return loss_c, loss_b, loss_m, loss_s

    def category_loss(self, class_p, conf_gt, pos_bool, np_ratio=3):
        # Compute max conf across batch for hard negative mining
        batch_conf = class_p.reshape(-1, self.cfg.num_classes)  # (38496, 81)

        batch_conf_max = batch_conf.max()
        mark = torch.log(torch.sum(torch.exp(batch_conf - batch_conf_max), 1)) + batch_conf_max - batch_conf[:, 0]

        # Hard Negative Mining
        mark = mark.reshape(class_p.size(0), -1)  # (n, 19248)
        mark[pos_bool] = 0  # filter out pos boxes
        mark[conf_gt < 0] = 0  # filter out neutrals (conf_gt = -1)

        _, idx = mark.sort(1, descending=True)
        _, idx_rank = idx.sort(1)

        num_pos = pos_bool.long().sum(1, keepdim=True)
        num_neg = torch.clamp(np_ratio * num_pos, max=pos_bool.size(1) - 1)
        neg_bool = idx_rank < num_neg.expand_as(idx_rank)

        # Just in case there aren't enough negatives, don't start using positives as negatives
        neg_bool[pos_bool] = 0
        neg_bool[conf_gt < 0] = 0  # Filter out neutrals

        # Confidence Loss Including Positive and Negative Examples
        class_p_mined = class_p[(pos_bool + neg_bool)].reshape(-1, self.cfg.num_classes)
        class_gt_mined = conf_gt[(pos_bool + neg_bool)]

        return self.cfg.conf_alpha * F.cross_entropy(class_p_mined, class_gt_mined, reduction='sum') / num_pos.sum()

    def box_loss(self, box_p, all_offsets, pos_bool):
        num_pos = pos_bool.sum()
        pos_box_p = box_p[pos_bool, :]
        pos_offsets = all_offsets[pos_bool, :]

        return self.cfg.bbox_alpha * F.smooth_l1_loss(pos_box_p, pos_offsets, reduction='sum') / num_pos

    def lincomb_mask_loss(self, pos_bool, anchor_max_i, coef_p, proto_p, mask_gt, anchor_max_gt):
        proto_h, proto_w = proto_p.shape[1:3]
        total_pos_num = pos_bool.sum()
        loss_m = 0
        for i in range(coef_p.size(0)):  # coef_p.shape: (n, 19248, 32)
            # downsample the gt mask to the size of 'proto_p'
            downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()  # (138, 138, num_objects)
            # binarize the gt mask because of the downsample operation
            downsampled_masks = downsampled_masks.gt(0.5).float()

            pos_anchor_i = anchor_max_i[i][pos_bool[i]]
            pos_anchor_box = anchor_max_gt[i][pos_bool[i]]
            pos_coef = coef_p[i][pos_bool[i]]

            if pos_anchor_i.size(0) == 0:
                continue

            # If exceeds the number of masks for training, select a random subset
            old_num_pos = pos_coef.size(0)
            if old_num_pos > self.cfg.masks_to_train:
                perm = torch.randperm(pos_coef.size(0))
                select = perm[:self.cfg.masks_to_train]
                pos_coef = pos_coef[select]
                pos_anchor_i = pos_anchor_i[select]
                pos_anchor_box = pos_anchor_box[select]

            num_pos = pos_coef.size(0)

            pos_mask_gt = downsampled_masks[:, :, pos_anchor_i]

            # mask assembly by linear combination
            # @ means dot product
            mask_p = torch.sigmoid(proto_p[i] @ pos_coef.t())  # mask_p.shape: (138, 138, num_pos)
            mask_p = crop(mask_p, pos_anchor_box)  # pos_anchor_box.shape: (num_pos, 4)
            # TODO: grad out of gt box is 0, should it be modified?
            # TODO: need an upsample before computing loss?
            mask_loss = F.binary_cross_entropy(torch.clamp(mask_p, 0, 1), pos_mask_gt, reduction='none')
            # aa = -pos_mask_gt*torch.log(mask_p) - (1-pos_mask_gt) * torch.log(1-mask_p)

            # Normalize the mask loss to emulate roi pooling's effect on loss.
            anchor_area = (pos_anchor_box[:, 2] - pos_anchor_box[:, 0]) * (pos_anchor_box[:, 3] - pos_anchor_box[:, 1])
            mask_loss = mask_loss.sum(dim=(0, 1)) / anchor_area

            if old_num_pos > num_pos:
                mask_loss *= old_num_pos / num_pos

            loss_m += torch.sum(mask_loss)

        return self.cfg.mask_alpha * loss_m / proto_h / proto_w / total_pos_num

    def semantic_seg_loss(self, segmentation_p, mask_gt, class_gt):
        # Note classes here exclude the background class, so num_classes = cfg.num_classes-1
        batch_size, num_classes, mask_h, mask_w = segmentation_p.size()  # (n, 80, 69, 69)
        loss_s = 0

        for i in range(batch_size):
            cur_segment = segmentation_p[i]
            cur_class_gt = class_gt[i]

            downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (mask_h, mask_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            downsampled_masks = downsampled_masks.gt(0.5).float()  # (num_objects, 69, 69)

            # Construct Semantic Segmentation
            segment_gt = torch.zeros_like(cur_segment, requires_grad=False)
            for j in range(downsampled_masks.size(0)):
                segment_gt[cur_class_gt[j]] = torch.max(segment_gt[cur_class_gt[j]], downsampled_masks[j])

            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_gt, reduction='sum')

        return self.cfg.semantic_alpha * loss_s / mask_h / mask_w / batch_size
