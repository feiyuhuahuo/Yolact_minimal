#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import glob
import torch
import os
import numpy as np
import pycocotools
import json
from terminaltables import AsciiTable
from collections import OrderedDict

from utils.box_utils import box_iou, mask_iou
from config import COCO_LABEL_MAP


class ProgressBar:
    def __init__(self, length, max_val):
        self.max_val = max_val
        self.length = length
        self.cur_val = 0

        self.cur_num_bars = -1
        self.update_str()

    def update_str(self):
        num_bars = int(self.length * (self.cur_val / self.max_val))

        if num_bars != self.cur_num_bars:
            self.cur_num_bars = num_bars
            self.string = '█' * num_bars + '░' * (self.length - num_bars)

    def get_bar(self, new_val):
        self.cur_val = new_val

        if self.cur_val > self.max_val:
            self.cur_val = self.max_val
        self.update_str()
        return self.string


def save_best(net, mask_map, cfg_name, step):
    weight = glob.glob('weights/best*')
    weight = [aa for aa in weight if cfg_name in aa]
    assert len(weight) <= 1, 'Error, multiple best weight found.'
    best_mask_map = float(weight[0].split('/')[-1].split('_')[1]) if weight else 0.

    if mask_map >= best_mask_map:
        if weight:
            os.remove(weight[0])  # remove the last best model

        print(f'\nSaving the best model as \'best_{mask_map}_{cfg_name}_{step}.pth\'.\n')
        torch.save(net.state_dict(), f'weights/best_{mask_map}_{cfg_name}_{step}.pth')


def save_latest(net, cfg_name, step):
    weight = glob.glob('weights/latest*')
    weight = [aa for aa in weight if cfg_name in aa]
    assert len(weight) <= 1, 'Error, multiple latest weight found.'
    if weight:
        os.remove(weight[0])

    print(f'\nSaving the latest model as \'latest_{cfg_name}_{step}.pth\'.\n')
    torch.save(net.state_dict(), f'weights/latest_{cfg_name}_{step}.pth')


class MakeJson:
    def __init__(self):
        self.bbox_data = []
        self.mask_data = []
        self.coco_cats = {}

        for coco_id, real_id in COCO_LABEL_MAP.items():
            class_id = real_id - 1
            self.coco_cats[class_id] = coco_id

    def add_bbox(self, image_id: int, category_id: int, bbox: list, score: float):
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x) * 10) / 10 for x in bbox]

        self.bbox_data.append({'image_id': int(image_id),
                               'category_id': self.coco_cats[int(category_id)],
                               'bbox': bbox,
                               'score': float(score)})

    def add_mask(self, image_id: int, category_id: int, segmentation: np.ndarray, score: float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')  # json.dump doesn't like bytes strings

        self.mask_data.append({'image_id': int(image_id),
                               'category_id': self.coco_cats[int(category_id)],
                               'segmentation': rle,
                               'score': float(score)})

    def dump(self):
        dump_arguments = [(self.bbox_data, f'results/bbox_detections.json'),
                          (self.mask_data, f'results/mask_detections.json')]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)


class APDataObject:
    """Stores all the information necessary to calculate the AP for one IoU and one class."""

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score: float, is_true: bool):
        self.data_points.append((score, is_true))

    def add_gt_positives(self, num_positives: int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls = []
        num_true = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]:
                num_true += 1
            else:
                num_false += 1

            precision = num_true / (num_true + num_false)
            recall = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions) - 1, 0, -1):
            if precisions[i] > precisions[i - 1]:
                precisions[i - 1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101  # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)


def prep_metrics(ap_data, ids_p, classes_p, boxes_p, masks_p, gt, gt_masks, height, width, iou_thres):
    gt_boxes = gt[:, :4]
    gt_boxes[:, [0, 2]] *= width
    gt_boxes[:, [1, 3]] *= height
    gt_classes = gt[:, 4].int().tolist()
    gt_masks = gt_masks.reshape(-1, height * width)
    masks_p = masks_p.reshape(-1, height * width)

    mask_iou_cache = mask_iou(masks_p, gt_masks)
    bbox_iou_cache = box_iou(boxes_p.float(), gt_boxes.float()).cpu()

    for _class in set(ids_p + gt_classes):
        num_gt_per_class = gt_classes.count(_class)

        for iouIdx in range(len(iou_thres)):
            iou_threshold = iou_thres[iouIdx]

            for iou_type, iou_func in zip(['box', 'mask'], [bbox_iou_cache, mask_iou_cache]):
                gt_used = [False] * len(gt_classes)
                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_per_class)

                for i, pred_class in enumerate(ids_p):
                    if pred_class != _class:
                        continue

                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j, gt_class in enumerate(gt_classes):
                        if gt_used[j] or gt_class != _class:
                            continue

                        iou = iou_func[i, j].item()

                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j

                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(classes_p[i], True)
                    else:
                        ap_obj.push(classes_p[i], False)


def calc_map(ap_data, iou_thres, num_classes, step):
    print('\nCalculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thres]

    for _class in range(num_classes):
        for iou_idx in range(len(iou_thres)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0  # Make this first in the ordereddict

        for i, threshold in enumerate(iou_thres):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold * 100)] = mAP

        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1))

    row1 = list(all_maps['box'].keys())
    row1.insert(0, f'{step // 1000}k' if step else '')

    row2 = list(all_maps['box'].values())
    row2 = [round(aa, 2) for aa in row2]
    row2.insert(0, 'box')

    row3 = list(all_maps['mask'].values())
    row3 = [round(aa, 2) for aa in row3]
    row3.insert(0, 'mask')

    table = [row1, row2, row3]
    table = AsciiTable(table)
    return table.table, row2, row3
