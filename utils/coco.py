import os.path as osp
import torch
import torch.utils.data as data
import cv2
import glob
import numpy as np
from pycocotools.coco import COCO

from utils.augmentations import train_aug, val_aug


def train_collate(batch):
    imgs, targets, masks, num_crowds = [], [], [], []
    valid_batch = [aa for aa in batch if aa[0] is not None]

    lack_len = len(batch) - len(valid_batch)
    if lack_len > 0:
        for i in range(lack_len):
            valid_batch.append(valid_batch[i])

    for sample in valid_batch:
        imgs.append(torch.tensor(sample[0], dtype=torch.float32))
        targets.append(torch.tensor(sample[1], dtype=torch.float32))
        masks.append(torch.tensor(sample[2], dtype=torch.float32))
        num_crowds.append(sample[3])

    return torch.stack(imgs, 0), targets, masks, num_crowds


def val_collate(batch):
    imgs = torch.tensor(batch[0][0], dtype=torch.float32).unsqueeze(0)
    targets = torch.tensor(batch[0][1], dtype=torch.float32)
    masks = torch.tensor(batch[0][2], dtype=torch.float32)
    return imgs, targets, masks, batch[0][3], batch[0][4], batch[0][5]


def detect_collate(batch):
    imgs = torch.tensor(batch[0][0], dtype=torch.float32).unsqueeze(0)
    return imgs, batch[0][1], batch[0][2]


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
            img_normed = val_aug(img_origin, self.cfg)
            return img_normed, img_origin, img_name.split('/')[-1]
        else:
            img_id = self.ids[index]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            # 'target' includes {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
            target = self.coco.loadAnns(ann_ids)

            # Separate out crowd annotations. When val and training, treat these crowds as neutral.
            crowd = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
            target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
            num_crowds = len(crowd)

            # Ensure that all crowd annotations are at the end of the array.
            target += crowd
            file_name = self.coco.loadImgs(img_id)[0]['file_name']

            img_path = osp.join(self.image_path, file_name)
            assert osp.exists(img_path), f'Image path does not exist: {img_path}'

            img = cv2.imread(img_path)
            height, width, _ = img.shape

            assert len(target) > 0, 'No annotation in this image!'
            scale = np.array([width, height, width, height])
            box_list, mask_list = [], []

            for aa in target:
                bbox = aa['bbox']

                # When training, some boxes are wrong, ignore them.
                if self.mode == 'train':
                    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] < 4 or bbox[3] < 4:
                        continue

                x1y1x2y2_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                final_box = list(x1y1x2y2_box / scale)
                category = self.continuous_id[aa['category_id']] - 1
                final_box.append(category)

                box_list.append(final_box)  # (xmin, ymin, xmax, ymax, label_idx), between 0~1
                mask_list.append(self.coco.annToMask(aa))
                # Uncomment this to visualize the masks.
                # cv2.imshow('aa', mask_list[0]*255)
                # cv2.waitKey()

            if len(box_list) > 0:
                box_array = np.array(box_list)
                masks = np.stack(mask_list, axis=0)

                assert masks.shape == (box_array.shape[0], height, width), 'Unmatched annotations.'

                boxes, labels = box_array[:, :4], box_array[:, 4]
                if self.mode == 'train':
                    img, masks, boxes, labels, num_crowds = train_aug(img, masks, boxes, labels, num_crowds, self.cfg)
                elif self.mode == 'val':
                    img = val_aug(img, self.cfg)

                boxes = np.hstack((boxes, np.expand_dims(labels, axis=1)))

                if self.mode == 'val':
                    return img, boxes, masks, num_crowds, height, width
                else:
                    return img, boxes, masks, num_crowds
            else:
                if self.mode == 'val':
                    raise RuntimeError('Error, no valid object in this image.')
                else:
                    print(f'Warning, no valid object in image: {img_id}. Use a repeated image in this batch.')
                    return None, None, None, None

    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        elif self.mode == 'val':
            return len(self.ids) if self.cfg.val_num == -1 else min(self.cfg.val_num, len(self.ids))
        elif self.mode == 'detect':
            return len(self.image_path)
