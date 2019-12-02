# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, center_size, crop
from data.config import cfg


class Multi_Loss(nn.Module):
    def __init__(self, num_classes, pos_thre, neg_thre, np_ratio):
        super().__init__()
        self.num_classes = num_classes
        self.pos_thre = pos_thre
        self.neg_thre = neg_thre
        self.negpos_ratio = np_ratio

    def ohem_conf_loss(self, class_p, conf_gt, positive_bool):
        # Compute max conf across batch for hard negative mining
        batch_conf = class_p.view(-1, self.num_classes)  # (38496, 81)

        batch_conf_max = batch_conf.data.max()
        mark = torch.log(torch.sum(torch.exp(batch_conf - batch_conf_max), 1)) + batch_conf_max - batch_conf[:, 0]

        # Hard Negative Mining
        mark = mark.view(class_p.size(0), -1)  # (n, 19248)
        mark[positive_bool] = 0  # filter out pos boxes
        mark[conf_gt < 0] = 0  # filter out neutrals (conf_gt = -1)

        _, idx = mark.sort(1, descending=True)
        _, idx_rank = idx.sort(1)

        num_pos = positive_bool.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=positive_bool.size(1) - 1)
        negative_bool = idx_rank < num_neg.expand_as(idx_rank)

        # Just in case there aren't enough negatives, don't start using positives as negatives
        negative_bool[positive_bool] = 0
        negative_bool[conf_gt < 0] = 0  # Filter out neutrals

        # Confidence Loss Including Positive and Negative Examples
        class_p_selected = class_p[(positive_bool + negative_bool)].view(-1, self.num_classes)
        class_gt_selected = conf_gt[(positive_bool + negative_bool)]

        loss_c = F.cross_entropy(class_p_selected, class_gt_selected, reduction='sum')

        return cfg.conf_alpha * loss_c

    @staticmethod
    def bbox_loss(pos_box_p, pos_offsets):
        loss_b = F.smooth_l1_loss(pos_box_p, pos_offsets, reduction='sum') * cfg.bbox_alpha
        return loss_b

    @staticmethod
    def lincomb_mask_loss(positive_bool, prior_max_index, coef_p, proto_p, mask_gt, prior_max_box):
        proto_h = proto_p.size(1)  # 138
        proto_w = proto_p.size(2)  # 138

        loss_m = 0
        for i in range(coef_p.size(0)):  # coef_p.shape: (n, 19248, 32)
            with torch.no_grad():
                # downsample the gt mask to the size of 'proto_p'
                downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (proto_h, proto_w), mode='bilinear',
                                                  align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.permute(1, 2, 0).contiguous()  # (138, 138, num_objects)
                # binarize the gt mask because of the downsample operation
                downsampled_masks = downsampled_masks.gt(0.5).float()

            pos_prior_index = prior_max_index[i, positive_bool[i]]  # pos_prior_index.shape: [num_positives]
            pos_prior_box = prior_max_box[i, positive_bool[i]]
            pos_coef = coef_p[i, positive_bool[i]]

            if pos_prior_index.size(0) == 0:
                continue

            # If exceeds the number of masks for training, select a random subset
            old_num_pos = pos_coef.size(0)
            if old_num_pos > cfg.masks_to_train:
                perm = torch.randperm(pos_coef.size(0))
                select = perm[:cfg.masks_to_train]
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
            pos_get_csize = center_size(pos_prior_box)
            mask_loss = mask_loss.sum(dim=(0, 1)) / pos_get_csize[:, 2] / pos_get_csize[:, 3]

            if old_num_pos > num_pos:
                mask_loss *= old_num_pos / num_pos

            loss_m += torch.sum(mask_loss)

        loss_m *= cfg.mask_alpha / proto_h / proto_w

        return loss_m

    @staticmethod
    def semantic_segmentation_loss(segmentation_p, mask_gt, class_gt):
        # Note classes here exclude the background class, so num_classes = cfg.num_classes-1
        batch_size, num_classes, mask_h, mask_w = segmentation_p.size()  # (n, 80, 69, 69)
        loss_s = 0

        for i in range(batch_size):
            cur_segment = segmentation_p[i]
            cur_class_gt = class_gt[i]

            with torch.no_grad():
                downsampled_masks = F.interpolate(mask_gt[i].unsqueeze(0), (mask_h, mask_w), mode='bilinear',
                                                  align_corners=False).squeeze(0)
                downsampled_masks = downsampled_masks.gt(0.5).float()  # (num_objects, 69, 69)

                # Construct Semantic Segmentation
                segment_gt = torch.zeros_like(cur_segment, requires_grad=False)
                for i_obj in range(downsampled_masks.size(0)):
                    segment_gt[cur_class_gt[i_obj]] = torch.max(segment_gt[cur_class_gt[i_obj]],
                                                                downsampled_masks[i_obj])

            loss_s += F.binary_cross_entropy_with_logits(cur_segment, segment_gt, reduction='sum')

        return loss_s / mask_h / mask_w * cfg.semantic_alpha

    def forward(self, predictions, box_class, mask_gt, num_crowds):
        # If DataParallel was used, predictions here are the merged results from multiple GPUs.
        box_p = predictions['box']  # (n, 19248, 4)
        class_p = predictions['class']  # (n, 19248, 81)
        coef_p = predictions['coef']  # (n, 19248, 32)
        anchors = predictions['anchors']  # (19248, 4)
        proto_p = predictions['proto']  # (n, 138, 138, 32)

        class_gt = [None] * len(box_class)
        batch_size = box_p.size(0)
        anchors = anchors[:box_p.size(1), :]  # Do this because DataParallel cats all GPU's anchors together.
        num_priors = (anchors.size(0))  # 19248

        all_offsets = box_p.new(batch_size, num_priors, 4)
        conf_gt = box_p.new(batch_size, num_priors).long()
        prior_max_box = box_p.new(batch_size, num_priors, 4)
        prior_max_index = box_p.new(batch_size, num_priors).long()

        for i in range(batch_size):
            box_gt = box_class[i][:, :-1].data
            class_gt[i] = box_class[i][:, -1].data.long()

            cur_crowds = num_crowds[i]
            if cur_crowds > 0:
                split = lambda x: (x[-cur_crowds:], x[:-cur_crowds])
                # drop the iscrowd boxes and masks
                crowd_boxes, box_gt = split(box_gt)
                _, class_gt[i] = split(class_gt[i])
                _, mask_gt[i] = split(mask_gt[i])
            else:
                crowd_boxes = None

            all_offsets[i], conf_gt[i], prior_max_box[i], prior_max_index[i] = match(self.pos_thre, self.neg_thre,
                                                                                     box_gt,
                                                                                     anchors, class_gt[i], crowd_boxes)

        # all_offsets: the transformed box coordinate offsets of each pair of prior and gt box
        # conf_gt: the foreground and background labels according to the 'pos_thre' and 'neg_thre',
        #          '0' means background, '>0' means foreground.
        # prior_max_box: the corresponding max IoU gt box for each prior
        # prior_max_index: the index of the corresponding max IoU gt box for each prior
        all_offsets = Variable(all_offsets, requires_grad=False)  # (n, 19248, 4)
        conf_gt = Variable(conf_gt, requires_grad=False)  # (n, 19248)
        prior_max_index = Variable(prior_max_index, requires_grad=False)  # (n, 19248)

        losses = {}

        # only compute losses from positive samples
        positive_bool = conf_gt > 0  # (n, 19248)
        num_pos = positive_bool.sum(dim=1, keepdim=True)

        pos_box_p = box_p[positive_bool, :]
        pos_offsets = all_offsets[positive_bool, :]

        losses['B'] = self.bbox_loss(pos_box_p, pos_offsets)
        losses['M'] = self.lincomb_mask_loss(positive_bool, prior_max_index, coef_p, proto_p, mask_gt, prior_max_box)
        losses['C'] = self.ohem_conf_loss(class_p, conf_gt, positive_bool)
        if cfg.train_semantic:
            losses['S'] = self.semantic_segmentation_loss(predictions['segm'], mask_gt, class_gt)

        total_num_pos = num_pos.data.sum().float()
        for aa in losses:
            if aa != 'S':
                losses[aa] /= total_num_pos
            else:
                losses[aa] /= batch_size

        return losses
