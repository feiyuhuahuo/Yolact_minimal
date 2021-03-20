import os.path as osp
import torch
import torch.utils.data as data
import cv2
import glob
import numpy as np
from pycocotools.coco import COCO

from utils.augmentations import train_aug, val_aug


def train_collate(batch):
    imgs, targets, masks = [], [], []
    valid_batch = [aa for aa in batch if aa[0] is not None]

    lack_len = len(batch) - len(valid_batch)
    if lack_len > 0:
        for i in range(lack_len):
            valid_batch.append(valid_batch[i])

    for sample in valid_batch:
        imgs.append(torch.tensor(sample[0], dtype=torch.float32))
        targets.append(torch.tensor(sample[1], dtype=torch.float32))
        masks.append(torch.tensor(sample[2], dtype=torch.float32))

    return torch.stack(imgs, 0), targets, masks


def val_collate(batch):
    imgs = torch.tensor(batch[0][0], dtype=torch.float32).unsqueeze(0)
    targets = torch.tensor(batch[0][1], dtype=torch.float32)
    masks = torch.tensor(batch[0][2], dtype=torch.float32)
    return imgs, targets, masks, batch[0][3], batch[0][4]


def detect_collate(batch):
    imgs = torch.tensor(batch[0][0], dtype=torch.float32).unsqueeze(0)
    return imgs, batch[0][1], batch[0][2]


def detect_onnx_collate(batch):
    return batch[0][0][None, :], batch[0][1], batch[0][2]


class COCODetection(data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.mode = mode
        self.cfg = cfg

        if mode in ('train', 'val'):
            self.image_path = cfg.train_imgs if mode == 'train' else cfg.val_imgs
            self.coco = COCO(cfg.train_ann if mode == 'train' else cfg.val_ann)
            self.ids = list(self.coco.imgToAnns.keys())
        elif mode == 'detect':
            self.image_path = glob.glob(cfg.image + '/*.jpg')
            self.image_path.sort()

        self.continuous_id = cfg.continuous_id

    def __getitem__(self, index):
        if self.mode == 'detect':
            img_name = self.image_path[index]
            img_origin = cv2.imread(img_name)
            img_normed = val_aug(img_origin, self.cfg.img_size)
            return img_normed, img_origin, img_name.split('/')[-1]
        else:
            img_id = self.ids[index]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            # 'target' includes {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = self.coco.loadAnns(ann_ids)
            target = [aa for aa in target if not aa['iscrowd']]

            file_name = self.coco.loadImgs(img_id)[0]['file_name']

            img_path = osp.join(self.image_path, file_name)
            assert osp.exists(img_path), f'Image path does not exist: {img_path}'

            img = cv2.imread(img_path)
            height, width, _ = img.shape

            assert len(target) > 0, 'No annotation in this image!'
            box_list, mask_list, label_list = [], [], []

            for aa in target:
                bbox = aa['bbox']

                # When training, some boxes are wrong, ignore them.
                if self.mode == 'train':
                    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 4 or bbox[3] < 4:
                        continue

                x1y1x2y2_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                category = self.continuous_id[aa['category_id']] - 1

                box_list.append(x1y1x2y2_box)
                mask_list.append(self.coco.annToMask(aa))
                label_list.append(category)

            if len(box_list) > 0:
                boxes = np.array(box_list)
                masks = np.stack(mask_list, axis=0)
                labels = np.array(label_list)
                assert masks.shape == (boxes.shape[0], height, width), 'Unmatched annotations.'

                if self.mode == 'train':
                    img, masks, boxes, labels = train_aug(img, masks, boxes, labels, self.cfg.img_size)
                    if img is None:
                        return None, None, None
                    else:
                        boxes = np.hstack((boxes, np.expand_dims(labels, axis=1)))
                        return img, boxes, masks
                elif self.mode == 'val':
                    img = val_aug(img, self.cfg.img_size)
                    boxes = boxes / np.array([width, height, width, height])  # to 0~1 scale
                    boxes = np.hstack((boxes, np.expand_dims(labels, axis=1)))
                    return img, boxes, masks, height, width
            else:
                if self.mode == 'val':
                    raise RuntimeError('Error, no valid object in this image.')
                else:
                    print(f'No valid object in image: {img_id}. Use a repeated image in this batch.')
                    return None, None, None

    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        elif self.mode == 'val':
            return len(self.ids) if self.cfg.val_num == -1 else min(self.cfg.val_num, len(self.ids))
        elif self.mode == 'detect':
            return len(self.image_path)
