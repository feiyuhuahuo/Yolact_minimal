import cv2
import numpy as np
import random

from config import COLORS
from config import norm_mean, norm_std
import pdb


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter

    return inter / union  # [A,B]


def to_img_scale_box(hw, boxes):
    boxes[:, [0, 2]] *= hw[1]
    boxes[:, [1, 3]] *= hw[0]
    return boxes


def to_01_box(hw, boxes):
    boxes[:, [0, 2]] /= hw[1]
    boxes[:, [1, 3]] /= hw[0]
    return boxes


# Not used currently.
def Pad(image, masks, boxes, labels, width, height, pad_gt=True):
    """
    Pads the image to the input width and height, filling the
    background with mean and putting the image in the top-left.

    Note: this expects im_w <= width and im_h <= height
    """
    width = width
    height = height
    pad_gt = pad_gt

    im_h, im_w, depth = image.shape

    expand_image = np.zeros((height, width, depth), dtype=image.dtype)
    expand_image[:, :, :] = norm_mean
    expand_image[:im_h, :im_w] = image

    if pad_gt:
        expand_masks = np.zeros((masks.shape[0], height, width), dtype=masks.dtype)
        expand_masks[:, :im_h, :im_w] = masks
        masks = expand_masks

    return expand_image, masks, boxes, labels


# TODO: try to use a keep ratio resize, also pay attention to square anchor
def resize(image, masks, boxes, img_size, during_training=True):
    img_h, img_w, _ = image.shape
    width, height = img_size, img_size
    image = cv2.resize(image, (width, height))

    if during_training:
        # Act like each object is a color channel
        masks = masks.transpose((1, 2, 0))
        masks = cv2.resize(masks, (width, height))

        # OpenCV resizes a (w,h,1) array to (s,s), so fix that
        if len(masks.shape) == 2:
            masks = np.expand_dims(masks, 0)
        else:
            masks = masks.transpose((2, 0, 1))

        # Scale bounding boxes (which are currently absolute coordinates)
        boxes[:, [0, 2]] *= (width / img_w)
        boxes[:, [1, 3]] *= (height / img_h)

    return image, masks, boxes


def random_brightness(image, delta=32):
    image += random.uniform(-delta, delta)

    return np.clip(image, 0., 255.)


def random_contrast(image, lower=0.7, upper=1.3):
    image *= random.uniform(lower, upper)

    return np.clip(image, 0., 255.)


def random_saturation(image, lower=0.7, upper=1.3):
    image[:, :, 1] *= random.uniform(lower, upper)
    return image


def random_hue(image, delta=15.):  # better keep 0.< delta <=30., two large delta harms the image
    image[:, :, 0] += random.uniform(-delta, delta)
    image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
    image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
    return image


def photometric_distort(image):
    if random.randint(0, 1):
        image = random_brightness(image)
    if random.randint(0, 1):
        image = random_contrast(image)

    # Because of normalization, random brightness and random contrast are meanless
    # if random_saturation and random_hue do not follow.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = random_saturation(image)
    image = random_hue(image)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    image = np.clip(image, 0., 255.)

    # TODO: need a random noise aug
    return image


# Potentialy sample a random crop from the image and put it in a random place
def random_crop(image, masks, boxes, labels):
    sample_options = (None,  # using entire original input image
                      (0.1, None),  # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
                      (0.3, None),
                      (0.7, None),
                      (0.9, None),
                      # randomly sample a patch
                      (None, None))

    height, width, _ = image.shape
    while True:
        mode = random.choice(sample_options)
        if mode is None:
            return image, masks, boxes, labels
        else:
            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(1, width - w)
                top = random.uniform(1, height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # This piece of code is bugged and does nothing: https://github.com/amdegroot/ssd.pytorch/issues/68
                #
                # However, when I fixed it with overlap.max() < min_iou,
                # it cut the mAP in half (after 8k iterations). So it stays.
                #
                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # [0 ... 0 for num_gt and then 1 ... 1 for num_crowds]
                crowd_mask = np.zeros(mask.shape, dtype=np.int32)

                # have any valid boxes? try again if not
                # Also make sure you have at least one regular gt
                if not mask.any() or np.sum(1 - crowd_mask[mask]) == 0:
                    continue

                # take only the matching gt masks
                current_masks = masks[mask, :, :].copy()

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                # crop the current masks to the same dimensions as the image
                current_masks = current_masks[:, rect[1]:rect[3], rect[0]:rect[2]]

                return current_image, current_masks, current_boxes, current_labels


def random_expand(image, masks, boxes):
    # TODO: now is only random reduce the original image, this benefits small object detection, also need a
    #  random enlarge
    if random.randint(0, 1):
        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(1, width * ratio - width)
        top = random.uniform(1, height * ratio - height)

        expand_image = np.zeros((int(height * ratio), int(width * ratio), depth), dtype=image.dtype)
        expand_image[:, :, :] = norm_mean
        expand_image[int(top):int(top + height), int(left):int(left + width)] = image
        image = expand_image

        expand_masks = np.zeros((masks.shape[0], int(height * ratio), int(width * ratio)), dtype=masks.dtype)
        expand_masks[:, int(top):int(top + height), int(left):int(left + width)] = masks
        masks = expand_masks

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

    return image, masks, boxes


def random_mirror(image, masks, boxes):
    if random.randint(0, 1):
        _, width, _ = image.shape
        image = image[:, ::-1]
        masks = masks[:, :, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]

    return image, masks, boxes


def normalize_and_toRGB(img):
    img = (img - norm_mean) / norm_std
    img = img[:, :, (2, 1, 0)]
    img = np.transpose(img, (2, 0, 1))
    return img


def val_aug(img, cfg):
    img, _, _ = resize(img, None, None, cfg.img_size, during_training=False)
    img = img.astype('float32')
    img = normalize_and_toRGB(img)
    return img


def train_aug(img, masks, boxes, labels, cfg):
    img = img.astype('float32')
    img = photometric_distort(img)
    boxes = to_img_scale_box(img.shape[:2], boxes)
    img, masks, boxes = random_expand(img, masks, boxes)
    # TODO: crop should be in front of expand maybe, crop should be implemented better, but before that,
    #  crowd anns issue should be figured out first, because crop deals with crowd anns.
    img, masks, boxes, labels = random_crop(img, masks, boxes, labels)
    img, masks, boxes = random_mirror(img, masks, boxes)
    img, masks, boxes = resize(img, masks, boxes, cfg.img_size, during_training=True)

    # masks_semantic = masks * (labels[:, None, None] + 1)  # expand class_ids' shape for broadcasting
    # # The color of the overlap area is different because of the '%' operation.
    # masks_semantic = masks_semantic.astype('int').sum(axis=0) % (cfg.num_classes - 1)
    # color_masks = COLORS[masks_semantic].astype('uint8')
    # img_u8 = img.astype('uint8')
    # img_fused = cv2.addWeighted(color_masks, 0.4, img_u8, 0.6, gamma=0)
    #
    # for i in range(boxes.shape[0]):
    #     cv2.rectangle(img_fused, (int(boxes[i, 0]), int(boxes[i, 1])),
    #                   (int(boxes[i, 2]), int(boxes[i, 3])), (0, 255, 0), 1)
    #
    # print(labels)
    # cv2.imshow('aa', img_fused)
    # cv2.waitKey()

    boxes = to_01_box(img.shape[:2], boxes)
    img = normalize_and_toRGB(img)

    return img, masks, boxes, labels
