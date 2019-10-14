import os.path as osp
import torch
import torch.utils.data as data
import cv2
import numpy as np
from .config import COCO_LABEL_MAP
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
    """MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>_ Dataset."""

    def __init__(self, image_path, info_file, augmentation=None):
        self.image_path = image_path
        self.coco = COCO(info_file)

        self.ids = list(self.coco.imgToAnns.keys())

        if len(self.ids) == 0:
            self.ids = list(self.coco.imgs.keys())

        self.augmentation = augmentation

    @staticmethod
    def annotation_transform(target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = COCO_LABEL_MAP[obj['category_id']] - 1
                final_box = list(np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) / scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("No bbox found for object ", obj)

        return res

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, masks, h, w, num_crowds = self.pull_item(index)
        return im, gt, masks, num_crowds

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        # 'target' includes {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
        target = self.coco.loadAnns(ann_ids)

        # Separate out crowd annotations. These are annotations that signify a large crowd of objects,
        # where there is no annotation for each individual object. When testing and training, treat these crowds as neutral.
        crowd = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        # Ensure that all crowd annotations are at the end of the array.
        target += crowd

        # The split here is to have compatibility with both COCO2014 and 2017 annotations.
        # In 2014, images have the pattern COCO_{train/val}2014_%012d.jpg, while in 2017 it's %012d.jpg.
        # Our script downloads the images as %012d.jpg so convert accordingly.
        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]

        path = osp.join(self.image_path, file_name)
        assert osp.exists(path), 'Image path does not exist: {}'.format(path)

        img = cv2.imread(path)
        height, width, _ = img.shape

        if len(target) > 0:
            # Pool all the masks for this image into one [num_objects, height, width] matrix
            masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
            masks = np.vstack(masks)
            masks = masks.reshape(-1, height, width)

        if len(target) > 0:
            target = self.annotation_transform(target, width, height)

        if self.augmentation is not None:
            if len(target) > 0:
                target = np.array(target)
                img, masks, boxes, labels = self.augmentation(img, masks, target[:, :4],
                                                              {'num_crowds': num_crowds, 'labels': target[:, 4]})

                # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
                num_crowds = labels['num_crowds']
                labels = labels['labels']

                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            else:
                img, _, _, _ = self.augmentation(img, np.zeros((1, height, width), dtype=np.float),
                                                 np.array([[0, 0, 1, 1]]),
                                                 {'num_crowds': 0, 'labels': np.array([0])})
                masks = None
                target = None

        return torch.from_numpy(img).permute(2, 0, 1), target, masks, height, width, num_crowds
