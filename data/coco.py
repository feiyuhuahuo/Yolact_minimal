import os.path as osp
import torch
import torch.utils.data as data
import cv2
import numpy as np
from data.config import cfg
from pycocotools.coco import COCO


def detection_collate(batch):
    imgs = []
    targets = []
    masks = []
    num_crowds = []

    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        masks.append(torch.FloatTensor(sample[2]))
        num_crowds.append(sample[3])

    return torch.stack(imgs, 0), targets, masks, num_crowds


class COCODetection(data.Dataset):
    def __init__(self, image_path, info_file, augmentation=None):
        self.image_path = image_path
        self.coco = COCO(info_file)
        self.ids = list(self.coco.imgToAnns.keys())
        self.augmentation = augmentation
        self.label_map = cfg.label_map

    def __getitem__(self, index):
        im, gt, masks, h, w, num_crowds = self.pull_item(index)
        return im, gt, masks, num_crowds

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_ids = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_ids)

        # 'target' includes {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
        target = self.coco.loadAnns(ann_ids)

        # Separate out crowd annotations. These are annotations that signify a large crowd of objects, where there is
        # no annotation for each individual object. When testing and training, treat these crowds as neutral.
        crowd = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        # Ensure that all crowd annotations are at the end of the array.
        target += crowd
        file_name = self.coco.loadImgs(img_ids)[0]['file_name']

        img_path = osp.join(self.image_path, file_name)
        assert osp.exists(img_path), f'Image path does not exist: {img_path}'

        img = cv2.imread(img_path)
        height, width, _ = img.shape

        if len(target) > 0:
            masks = [self.coco.annToMask(aa).reshape(-1) for aa in target]
            masks = np.vstack(masks)
            masks = masks.reshape((-1, height, width))  # between 0~1, (num_objs, height, width)
            # Uncomment this to visualize the masks.
            # cv2.imshow('aa', masks[0]*255)
            # cv2.waitKey()

            scale = np.array([width, height, width, height])
            box_list = []
            for obj in target:
                if 'bbox' in obj:
                    bbox = obj['bbox']
                    label_idx = self.label_map[obj['category_id']] - 1
                    final_box = list(np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) / scale)
                    final_box.append(label_idx)
                    box_list += [final_box]  # (xmin, ymin, xmax, ymax, label_idx), between 0~1
                else:
                    print("No bbox found for object ", obj)

        if self.augmentation is not None:
            if len(box_list) > 0:
                box_array = np.array(box_list)
                img, masks, boxes, labels = self.augmentation(img, masks, box_array[:, :4],
                                                              {'num_crowds': num_crowds, 'labels': box_array[:, 4]})

                # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations.
                num_crowds = labels['num_crowds']
                labels = labels['labels']
                boxes = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), boxes, masks, height, width, num_crowds
