#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from data import cfg
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

bbox_det_file = 'results/bbox_detections.json'
mask_det_file = 'results/mask_detections.json'
annotation_path = cfg.dataset.valid_info

print('Loading annotations...')
gt_annotations = COCO(annotation_path)

bbox_dets = gt_annotations.loadRes(bbox_det_file)
mask_dets = gt_annotations.loadRes(mask_det_file)

print('\nEvaluating BBoxes:')
bbox_eval = COCOeval(gt_annotations, bbox_dets, 'bbox')
bbox_eval.evaluate()
bbox_eval.accumulate()
bbox_eval.summarize()

print('\nEvaluating Masks:')
bbox_eval = COCOeval(gt_annotations, mask_dets, 'segm')
bbox_eval.evaluate()
bbox_eval.accumulate()
bbox_eval.summarize()
