# -*- coding: utf-8 -*-
import torch
from itertools import product
from math import sqrt
from data.config import cfg


def center_size(boxes):
    """ Convert prior_boxes to format: (cx, cy, w, h)."""
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]), 1)


def intersect(box_a, box_b):
    """
    We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n, A, 4].
      box_b: (tensor) bounding boxes, Shape: [n, B, 4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)

    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2), box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2), box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)

    return inter[:, :, :, 0] * inter[:, :, :, 1]


def jaccard(box_a, box_b, iscrowd: bool = False):
    """
    Compute the IoU of two sets of boxes.
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
        iscrowd: if True, put the crowd in box_b
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(
        inter)  # [A,B]
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(
        inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)


def match(pos_thresh, neg_thresh, box_gt, priors, class_gt, crowd_boxes):
    """
    Match each prior box with the ground truth box of the highest jaccard overlap, encode the bounding boxes,
    then return the matched indices corresponding to both confidence and location preds.
    Args:
        pos_thresh: (float) IoU > pos_thresh ==> positive.
        neg_thresh: (float) IoU < neg_thresh ==> negative.
        box_gt: (tensor) Ground truth boxes, Shape: [num_obj, 4], (xmin, ymin, xmax, ymax).
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors, 4], (center_x, center_y, w, h).
        class_gt: (tensor) All the class labels for the image, Shape: [num_obj].
        crowd_boxes: (tensor) All the crowd box annotations or None if there are none.

    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    priors = priors.data
    # Convert prior boxes to the form of [xmin, ymin, xmax, ymax].
    decoded_priors = torch.cat((priors[:, :2] - priors[:, 2:] / 2, priors[:, :2] + priors[:, 2:] / 2), 1)

    overlaps = jaccard(box_gt, decoded_priors)  # size: [num_objects, num_priors]

    each_box_max, each_box_index = overlaps.max(1)  # size [num_objects], the max IoU for each gt box
    each_prior_max, each_prior_index = overlaps.max(0)  # size [num_priors], the max IoU for each prior

    # For the max IoU prior for each gt box, set its IoU to 2. This ensures that it won't be filtered
    # in the threshold step even if the IoU is under the negative threshold. This is because that we want
    # at least one prior to match with each gt box or else we'd be wasting training data.
    each_prior_max.index_fill_(0, each_box_index, 2)

    # Set the index of the pair (prior, gt) we set the overlap for above.
    for j in range(each_box_index.size(0)):
        each_prior_index[each_box_index[j]] = j

    each_prior_box = box_gt[each_prior_index]  # size: [num_priors, 4]
    conf = class_gt[each_prior_index] + 1  # the class of the max IoU gt box for each prior, size: [num_priors]

    conf[each_prior_max < pos_thresh] = -1  # label as neutral
    conf[each_prior_max < neg_thresh] = 0  # label as background

    # Deal with crowd annotations for COCO
    if crowd_boxes is not None and cfg.crowd_iou_threshold < 1:
        # Size [num_priors, num_crowds]
        crowd_overlaps = jaccard(decoded_priors, crowd_boxes, iscrowd=True)
        # Size [num_priors]
        best_crowd_overlap, best_crowd_idx = crowd_overlaps.max(1)
        # Set non-positives with crowd iou of over the threshold to be neutral.
        conf[(conf <= 0) & (best_crowd_overlap > cfg.crowd_iou_threshold)] = -1

    offsets = encode(each_prior_box, priors)

    return offsets, conf, each_prior_box, each_prior_index


def make_anchors(conv_h, conv_w, scale):
    prior_data = []
    # Iteration order is important (it has to sync up with the convout)
    for j, i in product(range(conv_h), range(conv_w)):
        # + 0.5 because priors are in center
        x = (i + 0.5) / conv_w
        y = (j + 0.5) / conv_h

        for ar in cfg.aspect_ratios:
            ar = sqrt(ar)
            w = scale * ar / cfg.img_size
            h = scale / ar / cfg.img_size

            # This is for backward compatability with a bug where I made everything square by accident
            if cfg.use_square_anchors:  # True
                h = w

            prior_data += [x, y, w, h]

    return prior_data


def encode(matched, priors):
    variances = [0.1, 0.2]

    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]  # 10 * (Xg - Xa) / Wa
    g_cxcy /= (variances[0] * priors[:, 2:])  # 10 * (Yg - Ya) / Ha
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]  # 5 * log(Wg / Wa)
    g_wh = torch.log(g_wh) / variances[1]  # 5 * log(Hg / Ha)
    # return target for smooth_l1_loss
    offsets = torch.cat([g_cxcy, g_wh], 1)  # [num_priors, 4]

    return offsets


def decode(box_p, priors):
    """
    Decode predicted bbox coordinates using the same scheme
    employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf

        b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
        b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
        b_w = prior_w * exp(loc_w)
        b_h = prior_h * exp(loc_h)
    
    Note that loc is inputed as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
    while priors are inputed as [x, y, w, h] where each coordinate
    is relative to size of the image (even sigmoid(x)). We do this
    in the network by dividing by the 'cell size', which is just
    the size of the convouts.
    
    Also note that prior_x and prior_y are center coordinates which
    is why we have to subtract .5 from sigmoid(pred_x and pred_y).
    """
    variances = [0.1, 0.2]

    boxes = torch.cat((priors[:, :2] + box_p[:, :2] * variances[0] * priors[:, 2:],
                       priors[:, 2:] * torch.exp(box_p[:, 2:] * variances[1])), 1)

    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    return boxes


def sanitize_coordinates(_x1, _x2, img_size: int, padding: int = 0):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size

    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1 - padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)

    return x1, x2


def crop(masks, boxes, padding: int = 1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask.float()


def mask_iou(mask1, mask2, iscrowd=False):
    """
    Inputs inputs are matricies of size _ x N. Output is size _1 x _2.
    Note: if iscrowd is True, then mask2 should be the crowd.
    """
    intersection = torch.matmul(mask1, mask2.t())
    area1 = torch.sum(mask1, dim=1).view(1, -1)
    area2 = torch.sum(mask2, dim=1).view(1, -1)
    union = (area1.t() + area2) - intersection

    if iscrowd:
        # Make sure to brodcast to the right dimension
        ret = intersection / area1.t()
    else:
        ret = intersection / union

    return ret.cpu()


def bbox_iou(bbox1, bbox2, iscrowd=False):
    ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()
