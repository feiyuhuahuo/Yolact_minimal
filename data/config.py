from modules.backbone import ResNetBackbone, DarkNetBackbone

COLORS = ((244, 67, 54), (233, 30, 99), (156, 39, 176), (103, 58, 183),
          (63, 81, 181), (33, 150, 243), (3, 169, 244), (0, 188, 212),
          (0, 150, 136), (76, 175, 80), (139, 195, 74), (205, 220, 57),
          (255, 235, 59), (255, 193, 7), (255, 152, 0), (255, 87, 34),
          (121, 85, 72), (158, 158, 158), (96, 125, 139))

# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD = (57.38, 57.12, 58.40)

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                  9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}


class Config(object):
    """
    After implement this class, you can call 'cfg.x' instead of 'cfg['x']' to get a certain parameter.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making the changes given by new_config_dict.
        """
        ret = Config(vars(self))

        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object. Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)


# ----------------------- DATASETS ----------------------- #
dataset_base = Config({'name': 'COCO 2017',
                       'train_images': '/home/feiyu/Data/coco2017/train2017/',
                       'train_info': '/home/feiyu/Data/coco2017/annotations/instances_train2017.json',
                       'valid_images': '/home/feiyu/Data/coco2017/val2017/',
                       'valid_info': '/home/feiyu/Data/coco2017/annotations/instances_val2017.json',
                       'class_names': COCO_CLASSES})

# ----------------------- TRANSFORMS ----------------------- #
resnet_transform = Config({'channel_order': 'RGB',
                           'normalize': True,
                           'subtract_means': False,
                           'to_float': False})

darknet_transform = Config({'channel_order': 'RGB',
                            'normalize': False,
                            'subtract_means': False,
                            'to_float': True})

# ----------------------- BACKBONES ----------------------- #
resnet101_backbone = Config({'name': 'ResNet101',
                             'path': 'resnet101_reducedfc.pth',
                             'type': ResNetBackbone,
                             'args': ([3, 4, 23, 3],),
                             'transform': resnet_transform,
                             'selected_layers': [1, 2, 3],
                             'scales': [24, 48, 96, 192, 384],
                             'aspect_ratios': [1, 1 / 2, 2],
                             'use_square_anchors': True})  # This is for backward compatability with a bug.

resnet50_backbone = resnet101_backbone.copy({'name': 'ResNet50',
                                             'path': 'resnet50-19c8e357.pth',
                                             'args': ([3, 4, 6, 3],)})

darknet53_backbone = Config({'name': 'DarkNet53',
                             'path': 'darknet53.pth',
                             'type': DarkNetBackbone,
                             'args': ([1, 2, 8, 8, 4],),
                             'transform': darknet_transform,
                             'selected_layers': [2, 3, 4],
                             'scales': [24, 48, 96, 192, 384],
                             'aspect_ratios': [1, 1 / 2, 2],
                             'use_square_anchors': True})

# ----------------------- FPN DEFAULTS ----------------------- #
fpn_base = Config({
    'num_features': 256,  # the number of features in each FPN layer
    'interpolation_mode': 'bilinear',  # upsampling mode
    'num_downsample': 2,  # The number of extra layers to be produced by downsampling starting at P5
    # Whether to down sample with a 3x3 stride 2 conv layer instead of just a stride 2 selection
    'use_conv_downsample': True,
    # Whether to pad the pred layers with 1 on each side, this is just for backwards compatibility
    'pad': 1})

# ----------------------- YOLACT v1.0 CONFIGS ----------------------- #
yolact_base_config = Config({
    'name': 'yolact_base',
    'dataset': dataset_base,
    'num_classes': len(dataset_base.class_names) + 1,
    'max_size': 550,  # image Size
    'max_iter': 800000,
    'backbone': resnet101_backbone,
    'hws': [(69, 69), (35, 35), (18, 18), (9, 9), (5, 5)],
    # During training, first compute the maximum gt IoU for each prior.
    # Then, for priors whose maximum IoU is over the positive threshold, marked as positive.
    # For priors whose maximum IoU is less than the negative threshold, marked as negative.
    # The rest are neutral ones and are not used to calculate the loss.
    'positive_iou_threshold': 0.5,
    'negative_iou_threshold': 0.4,
    'bbox_alpha': 1.5,
    'conf_alpha': 1,
    # If less than 1, anchors treated as a negative that have a crowd iou over this threshold with
    # the crowd boxes will be treated as a neutral.
    'crowd_iou_threshold': 0.7,
    # FPN Settings
    'fpn': fpn_base,
    # Freeze the backbone bn layer during training, other additional bn layers after the backbone will not be frozen.
    'freeze_bn': False,

    # Learning rate
    'lr_steps': (280000, 600000, 700000, 750000),
    'lr': 1e-3,
    'momentum': 0.9,
    'decay': 5e-4,
    # warm up setting
    'warmup_init': 1e-4,
    'warmup_until': 500,

    # The maximum number of detections for evaluation
    'mask_alpha': 6.125,
    'mask_proto_net': [(256, 3, {'padding': 1}), (256, 3, {'padding': 1}), (256, 3, {'padding': 1}),
                       (None, -2, {}), (256, 3, {'padding': 1}), (32, 1, {})],
    'extra_head_net': [(256, 3, {'padding': 1})],
    'coef_dim': None,
    'masks_to_train': 100,

    # Whether to train the semantic segmentations branch, this branch is only implemented during training
    # and is just for multitask learning.
    'train_semantic': True,
    'semantic_alpha': 1,
    # postprocess hyperparameters
    'conf_thre': 0.05,
    'nms_thre': 0.5,
    'top_k': 200,
    'max_detections': 100})

yolact_im400_config = yolact_base_config.copy({
    'name': 'yolact_im400',
    'max_size': 400,
    'backbone': yolact_base_config.backbone.copy({'scales': [17, 34, 69, 139, 279]})
})

yolact_im700_config = yolact_base_config.copy({
    'name': 'yolact_im700',
    'masks_to_train': 300,
    'max_size': 700,
    'backbone': yolact_base_config.backbone.copy({'scales': [30, 61, 122, 244, 488]})
})

yolact_darknet53_config = yolact_base_config.copy({'name': 'yolact_darknet53',
                                                   'backbone': darknet53_backbone})

yolact_resnet50_config = yolact_base_config.copy({'name': 'yolact_resnet50',
                                                  'backbone': resnet50_backbone})


def set_cfg(config_name: str):
    """ Sets the active config. Works even if cfg is already imported! """
    global cfg
    cfg.replace(eval(config_name))

# Default config
cfg = yolact_base_config.copy()
