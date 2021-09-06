#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2
from pprint import pprint as pp
from pycocotools.coco import COCO

coco = COCO('/home/feiyu/Data/pe/seg_m3/划痕/custom_ann.json')
ids = list(coco.imgToAnns.keys())
cv2.namedWindow('aa', cv2.WINDOW_NORMAL)
cv2.resizeWindow('aa', 900, 900)
ann_ids = coco.getAnnIds(imgIds=8)
print(ann_ids)

target = coco.loadAnns(ann_ids)

for one in target:
    pp(one['bbox'])
    bb = coco.annToMask(one).astype('uint8') * 100

    cv2.imshow('aa', bb)
    cv2.waitKey()

