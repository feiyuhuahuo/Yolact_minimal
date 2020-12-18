import torch
import torch.nn as nn
import torch.nn.functional as F

from config import mask_proto_net, extra_head_net
from modules.backbone import construct_backbone
from utils.box_utils import make_anchors
from utils.box_utils import match, crop


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
        # create a ModuleList to match with the original pre-trained weights (original model state_dict)
        self.prediction_layers = nn.ModuleList()
        self.prediction_layers.append(PredictionModule(cfg, coef_dim=coef_dim))

        if cfg.mode == 'train':
            self.semantic_seg_conv = nn.Conv2d(256, cfg.num_classes - 1, kernel_size=1)

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
        '''
        outs:
        (n, 3, 550, 550) -> backbone -> (n, 256, 138, 138) -> fpn -> (n, 256, 69, 69) P3
                                        (n, 512, 69, 69)             (n, 256, 35, 35) P4
                                        (n, 1024, 35, 35)            (n, 256, 18, 18) P5
                                        (n, 2048, 18, 18)            (n, 256, 9, 9)   P6
                                                                     (n, 256, 5, 5)   P7
        '''

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

    def ohem_conf_loss(self, class_p, conf_gt, positive_bool, np_ratio=3):
        # Compute max conf across batch for hard negative mining
        batch_conf = class_p.reshape(-1, self.cfg.num_classes)  # (38496, 81)

        batch_conf_max = batch_conf.max()
        mark = torch.log(torch.sum(torch.exp(batch_conf - batch_conf_max), 1)) + batch_conf_max - batch_conf[:, 0]

        # Hard Negative Mining
        mark = mark.reshape(class_p.size(0), -1)  # (n, 19248)
        mark[positive_bool] = 0  # filter out pos boxes
        mark[conf_gt < 0] = 0  # filter out neutrals (conf_gt = -1)

        _, idx = mark.sort(1, descending=True)
        _, idx_rank = idx.sort(1)

        num_pos = positive_bool.long().sum(1, keepdim=True)
        num_neg = torch.clamp(np_ratio * num_pos, max=positive_bool.size(1) - 1)
        negative_bool = idx_rank < num_neg.expand_as(idx_rank)

        # Just in case there aren't enough negatives, don't start using positives as negatives
        negative_bool[positive_bool] = 0
        negative_bool[conf_gt < 0] = 0  # Filter out neutrals

        # Confidence Loss Including Positive and Negative Examples
        class_p_selected = class_p[(positive_bool + negative_bool)].reshape(-1, self.cfg.num_classes)
        class_gt_selected = conf_gt[(positive_bool + negative_bool)]

        return self.cfg.conf_alpha * F.cross_entropy(class_p_selected, class_gt_selected, reduction='sum')

    def lincomb_mask_loss(self, positive_bool, prior_max_index, coef_p, proto_p, mask_gt, prior_max_box):
        proto_h = proto_p.size(1)  # 138
        proto_w = proto_p.size(2)  # 138

        loss_m = 0
        for i in range(coef_p.size(0)):  # coef_p.shape: (n, 19248, 32)
            # downsample the gt mask to the size of 'proto_p'
            downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                              align_corners=False).squeeze(0)
            downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()  # (138, 138, num_objects)
            # binarize the gt mask because of the downsample operation
            downsampled_masks = downsampled_masks.gt(0.5).float()

            pos_prior_index = prior_max_index[i][positive_bool[i]]  # pos_prior_index.shape: [num_positives]
            pos_prior_box = prior_max_box[i][positive_bool[i]]
            pos_coef = coef_p[i][positive_bool[i]]

            if pos_prior_index.size(0) == 0:
                continue

            # If exceeds the number of masks for training, select a random subset
            old_num_pos = pos_coef.size(0)
            if old_num_pos > self.cfg.masks_to_train:
                perm = torch.randperm(pos_coef.size(0))
                select = perm[:self.cfg.masks_to_train]
                pos_coef = pos_coef[select]
                pos_prior_index = pos_prior_index[select]
                pos_prior_box = pos_prior_box[select]

            num_pos = pos_coef.size(0)
            pos_mask_gt = downsampled_masks[:, :, pos_prior_index]

            # mask assembly by linear combination
            # @ means dot product
            mask_p = torch.sigmoid(proto_p[i] @ pos_coef.t())  # mask_p.shape: (138, 138, num_pos)
            mask_p = crop(mask_p, pos_prior_box)  # pos_prior_box.shape: (num_pos, 4)

            mask_loss = F.binary_cross_entropy(torch.clamp(mask_p, 0, 1), pos_mask_gt, reduction='none')
            # Normalize the mask loss to emulate roi pooling's effect on loss.
            prior_area = (pos_prior_box[:, 2] - pos_prior_box[:, 0]) * (pos_prior_box[:, 3] - pos_prior_box[:, 1])
            mask_loss = mask_loss.sum(dim=(0, 1)) / prior_area

            if old_num_pos > num_pos:
                mask_loss *= old_num_pos / num_pos

            loss_m += torch.sum(mask_loss)

        loss_m *= self.cfg.mask_alpha / proto_h / proto_w

        return loss_m

    def semantic_segmentation_loss(self, segmentation_p, mask_gt, class_gt):
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

        return loss_s / mask_h / mask_w * self.cfg.semantic_alpha

    def compute_loss(self, class_p, box_p, coef_p, proto_p, seg_p, box_class, mask_gt):
        class_gt = [None] * len(box_class)
        batch_size = box_p.size(0)
        num_priors = self.anchors.size(0)

        all_offsets = box_p.new(batch_size, num_priors, 4)
        conf_gt = box_p.new(batch_size, num_priors).long()
        anchor_max_box = box_p.new(batch_size, num_priors, 4)
        anchor_max_i = box_p.new(batch_size, num_priors).long()

        for i in range(batch_size):
            box_gt = box_class[i][:, :-1]
            class_gt[i] = box_class[i][:, -1].long()

            all_offsets[i], conf_gt[i], anchor_max_box[i], anchor_max_i[i] = match(self.cfg, box_gt,
                                                                                   self.anchors, class_gt[i])

        # all_offsets: the transformed box coordinate offsets of each pair of anchor and gt box
        # conf_gt: the foreground and background labels according to the 'pos_thre' and 'neg_thre',
        #          '0' means background, '>0' means foreground.
        # anchor_max_box: the corresponding max IoU gt box for each prior
        # anchor_max_i: the index of the corresponding max IoU gt box for each prior
        assert (not all_offsets.requires_grad) and (not conf_gt.requires_grad) and (not anchor_max_i.requires_grad), \
            'Incorrect computation graph, check the grad.'

        # only compute losses from positive samples
        positive_bool = conf_gt > 0  # (n, 19248)
        num_pos = positive_bool.sum()

        pos_box_p = box_p[positive_bool, :]
        pos_offsets = all_offsets[positive_bool, :]

        loss_b = F.smooth_l1_loss(pos_box_p, pos_offsets, reduction='sum') * self.cfg.bbox_alpha
        loss_m = self.lincomb_mask_loss(positive_bool, anchor_max_i, coef_p, proto_p, mask_gt, anchor_max_box)
        loss_c = self.ohem_conf_loss(class_p, conf_gt, positive_bool)
        loss_s = self.semantic_segmentation_loss(seg_p, mask_gt, class_gt)

        loss_b /= num_pos
        loss_m /= num_pos
        loss_c /= num_pos
        loss_s /= batch_size

        return loss_b, loss_m, loss_c, loss_s
