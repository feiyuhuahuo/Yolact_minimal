#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
import glob
import json
import os.path as osp
import numpy as np
import labelme
import pycocotools.mask

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--img_dir', help='input annotated directory')
parser.add_argument('--label_name', help='labels file')
parser.add_argument('--img_type', default='jpg', help='jpg, png, bmp...')
args = parser.parse_args()

data = dict(images=[], annotations=[], categories=[])

class_name_to_id = {}
for i, line in enumerate(open(args.label_name).readlines()):
    class_name = line.strip()
    class_name_to_id[class_name] = i
    data['categories'].append(dict(id=i, name=class_name))

print(f'Created class_name_to_id: {class_name_to_id}.\n')
class_name_to_id_key = list(class_name_to_id.keys())

label_files = glob.glob(osp.join(args.img_dir, '*.json'))
for image_id, label_file in enumerate(label_files):
    print('Generating dataset from:', label_file)

    with open(label_file) as f:
        label_data = json.load(f)

    img_h, img_w = label_data['imageHeight'], label_data['imageWidth']
    data['images'].append(dict(file_name=label_file.split('/')[-1].replace('json', args.img_type),
                               height=img_h,
                               width=img_w,
                               id=image_id))

    for shape in label_data['shapes']:
        points = shape['points']
        label = shape['label']
        shape_type = shape.get('shape_type', None)
        mask = labelme.utils.shape_to_mask((img_h, img_w), points, shape_type)

        assert label in class_name_to_id_key, f'Error, {label} not in class_name_to_id.'
        cls_id = class_name_to_id[label]

        mask = np.asfortranarray(mask.astype(np.uint8))
        mask = pycocotools.mask.encode(mask)
        area = float(pycocotools.mask.area(mask))
        bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

        data['annotations'].append(dict(id=len(data['annotations']),
                                        image_id=image_id,
                                        category_id=cls_id,
                                        segmentation=[np.asarray(points).flatten().tolist()],
                                        area=area,
                                        bbox=bbox,
                                        iscrowd=0))

with open(osp.join(args.img_dir, 'custom_ann.json'), 'w') as f:
    json.dump(data, f)

print('Saved in: ' + osp.join(args.img_dir, 'custom_ann.json'))
