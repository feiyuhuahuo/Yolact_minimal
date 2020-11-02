#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
import collections
import datetime
import glob
import json
import os.path as osp
import numpy as np
import labelme
import pycocotools.mask

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img_dir', help='input annotated directory')
parser.add_argument('--label_name', help='labels file')
args = parser.parse_args()

now = datetime.datetime.now()
data = dict(info=dict(description=None,
                      url=None,
                      version=None,
                      year=now.year,
                      contributor=None,
                      date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f')),
            images=[],  # license, url, file_name, height, width, date_captured, id
            type='instances',
            annotations=[],  # segmentation, area, iscrowd, image_id, bbox, category_id, id
            categories=[])  # supercategory, id, name

class_name_to_id = {}
for i, line in enumerate(open(args.label_name).readlines()):
    class_name = line.strip()
    class_name_to_id[class_name] = i
    data['categories'].append(dict(id=i, name=class_name))

label_files = glob.glob(osp.join(args.img_dir, '*.json'))
for image_id, label_file in enumerate(label_files):
    print('Generating dataset from:', label_file)

    with open(label_file) as f:
        label_data = json.load(f)

    img_h, img_w = label_data['imageHeight'], label_data['imageWidth']
    data['images'].append(dict(file_name=label_file.split('/')[-1].replace('json', 'jpg'),
                               height=img_h,
                               width=img_w,
                               id=image_id))

    masks = {}  # for area
    segmentations = collections.defaultdict(list)  # for segmentation
    for shape in label_data['shapes']:
        points = shape['points']
        label = shape['label']
        shape_type = shape.get('shape_type', None)
        mask = labelme.utils.shape_to_mask((img_h, img_w), points, shape_type)

        if label in masks:
            masks[label] = masks[label] | mask
        else:
            masks[label] = mask

        points = np.asarray(points).flatten().tolist()
        segmentations[label].append(points)

    for label, mask in masks.items():
        cls_name = label.split('-')[0]
        if cls_name not in class_name_to_id:
            continue
        cls_id = class_name_to_id[cls_name]

        mask = np.asfortranarray(mask.astype(np.uint8))
        mask = pycocotools.mask.encode(mask)
        area = float(pycocotools.mask.area(mask))
        bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

        data['annotations'].append(dict(id=len(data['annotations']),
                                        image_id=image_id,
                                        category_id=cls_id,
                                        segmentation=segmentations[label],
                                        area=area,
                                        bbox=bbox,
                                        iscrowd=0))

with open(osp.join(args.img_dir, 'custom_ann.json'), 'w') as f:
    json.dump(data, f)
