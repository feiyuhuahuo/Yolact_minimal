import os.path as osp
import torch
import torch.utils.data as data
import cv2
import numpy as np
from pycocotools.coco import COCO
from utils.augmentations import TrainAug, ValAug


def detection_collate(batch):
    imgs, targets, masks, num_crowds = [], [], [], []
    valid_batch = [aa for aa in batch if aa[0] is not None]

    lack_len = len(batch) - len(valid_batch)
    if lack_len > 0:
        for i in range(lack_len):
            valid_batch.append(valid_batch[i])

    for sample in valid_batch:
        imgs.append(sample[0])
        targets.append(torch.tensor(sample[1], dtype=torch.float32))
        masks.append(torch.tensor(sample[2], dtype=torch.float32))
        num_crowds.append(sample[3])

    return torch.stack(imgs, 0), targets, masks, num_crowds


class COCODetection(data.Dataset):
    def __init__(self, cfg, val_mode):
        self.val_mode = val_mode
        self.image_path = cfg.train_imgs if not val_mode else cfg.val_imgs
        self.coco = COCO(cfg.train_ann if not val_mode else cfg.val_ann)
        self.ids = list(self.coco.imgToAnns.keys())
        self.augmentation = TrainAug(cfg) if not val_mode else ValAug(cfg)
        self.continuous_id = cfg.continuous_id

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        # 'target' includes {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
        target = self.coco.loadAnns(ann_ids)

        # Separate out crowd annotations. These are annotations that signify a large crowd of objects, where there is
        # no annotation for each individual object. When testing and training, treat these crowds as neutral.
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

            # Some boxes are too small, ignore them.
            if bbox[2] < 4 or bbox[3] < 4:
                continue

            x1y1x2y2_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

            if (x1y1x2y2_box[0] >= x1y1x2y2_box[2]) or (x1y1x2y2_box[1] >= x1y1x2y2_box[3]):
                continue

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
            img, masks, boxes, labels = self.augmentation(img, masks, box_array[:, :4],
                                                          {'num_crowds': num_crowds, 'labels': box_array[:, 4]})

            # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations.
            num_crowds = labels['num_crowds']
            labels = labels['labels']
            boxes = np.hstack((boxes, np.expand_dims(labels, axis=1)))

            if self.val_mode:
                return torch.from_numpy(img).permute(2, 0, 1), boxes, masks, height, width, num_crowds
            else:
                return torch.from_numpy(img).permute(2, 0, 1), boxes, masks, num_crowds
        else:
            if self.val_mode:
                print('Error, no valid object in this image.')
                exit()
            else:
                print(f'Warning, no valid object in image: {img_id}. Use a repeated image in this batch.')
                return None, None, None, None
