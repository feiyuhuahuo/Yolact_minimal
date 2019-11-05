#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import scipy.io
import cv2
import os.path
import json
import pycocotools.mask
import numpy as np
from data.config import pascal_sbd_dataset


def mask2bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return cmin, rmin, cmax - cmin, rmax - rmin


img_path = pascal_sbd_dataset.train_images
root_path = img_path.split('img')[0]
inst_path = root_path + '/inst'

img_name_fmt = '%s.jpg'
ann_name_fmt = '%s.mat'

image_id = 1
ann_id = 1

types = ['train', 'val']

for t in types:
    with open(root_path + f'/{t}.txt', 'r') as f:
        names = f.read().strip().split('\n')

    images = []
    annotations = []

    for i, name in enumerate(names):
        img_name = img_name_fmt % name
        ann_path = os.path.join(inst_path, ann_name_fmt % name)
        ann = scipy.io.loadmat(ann_path)['GTinst'][0][0]

        classes = [int(x[0]) for x in ann[2]]
        seg = ann[0]

        for idx in range(len(classes)):
            mask = (seg == (idx + 1)).astype(np.float)

            rle = pycocotools.mask.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('ascii')

            annotations.append({'id': ann_id,
                                'image_id': image_id,
                                'category_id': classes[idx],
                                'segmentation': rle,
                                'area': float(mask.sum()),
                                'bbox': [int(x) for x in mask2bbox(mask)],
                                'iscrowd': 0})

            ann_id += 1

        img_name = img_name_fmt % name
        img = cv2.imread(os.path.join(img_path, img_name))

        images.append({'id': image_id,
                       'width': img.shape[1],
                       'height': img.shape[0],
                       'file_name': img_name})

        image_id += 1
        print(f'\r{i}', end='')

    info = {'year': 2012,
            'version': 1,
            'description': 'Pascal SBD'}

    categories = [{'id': x + 1} for x in range(20)]

    with open(root_path + f'pascal_sbd_{t}.json', 'w') as f:
        json.dump({'info': info,
                   'images': images,
                   'annotations': annotations,
                   'licenses': {},
                   'categories': categories}, f)
