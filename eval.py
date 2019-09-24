from data.coco import COCODetection, COCO_LABEL_MAP
from modules.build_yolact import Yolact
from utils.augmentations import BaseTransform
from utils.functions import MovingAverage, ProgressBar
from utils.box_utils import jaccard
from utils import timer
from utils.output_utils import postprocess, NMS
import pycocotools
from data.config import cfg, set_cfg
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import json
import os
from terminaltables import AsciiTable
from collections import OrderedDict


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model', default='weights/yolact_base_54_800000.pth', type=str)
    parser.add_argument('--top_k', default=5, type=int, help='Further restrict the number of predictions to parse')
    parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images for test, set to -1 for all.')
    parser.add_argument('--output_coco_json', action='store_true', help='Dumps detections into the coco json file.')
    parser.add_argument('--config', default=None, help='The config object to use.')
    parser.add_argument('--benchmark', default=False, action='store_true', help='do benchmark')

    global args
    args = parser.parse_args(argv)


def prep_benchmark(dets_out, h, w):
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h)
    with timer.env('Copy'):
        classes, scores, boxes, masks = [x[:args.top_k].cpu().numpy() for x in t]
    with timer.env('Sync'):
        torch.cuda.synchronize()


class Make_json:
    def __init__(self):
        self.bbox_data = []
        self.mask_data = []
        self.coco_cats = {}

        for coco_id, real_id in COCO_LABEL_MAP.items():
            class_id = real_id - 1
            self.coco_cats[class_id] = coco_id

    def add_bbox(self, image_id: int, category_id: int, bbox: list, score: float):
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x) * 10) / 10 for x in bbox]

        self.bbox_data.append({'image_id': int(image_id),
                               'category_id': self.coco_cats[int(category_id)],
                               'bbox': bbox,
                               'score': float(score)})

    def add_mask(self, image_id: int, category_id: int, segmentation: np.ndarray, score: float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')  # json.dump doesn't like bytes strings

        self.mask_data.append({'image_id': int(image_id),
                               'category_id': self.coco_cats[int(category_id)],
                               'segmentation': rle,
                               'score': float(score)})

    def dump(self):
        dump_arguments = [(self.bbox_data, 'results/bbox_detections.json'),
                          (self.mask_data, 'results/mask_detections.json')]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)


def mask_iou(mask1, mask2, iscrowd=False):
    """
    Inputs inputs are matricies of size _ x N. Output is size _1 x _2.
    Note: if iscrowd is True, then mask2 should be the crowd.
    """
    timer.start('Mask IoU')

    intersection = torch.matmul(mask1, mask2.t())
    area1 = torch.sum(mask1, dim=1).view(1, -1)
    area2 = torch.sum(mask2, dim=1).view(1, -1)
    union = (area1.t() + area2) - intersection

    if iscrowd:
        # Make sure to brodcast to the right dimension
        ret = intersection / area1.t()
    else:
        ret = intersection / union
    timer.stop('Mask IoU')

    return ret.cpu()


def bbox_iou(bbox1, bbox2, iscrowd=False):
    with timer.env('BBox IoU'):
        ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()


def prep_metrics(ap_data, dets, gt, gt_masks, h, w, num_crowd, image_id, make_json, coco_json):
    """ Returns a list of APs for this image, with each element being for a class  """
    if not coco_json:
        with timer.env('Prepare gt'):
            gt_boxes = torch.Tensor(gt[:, :4])
            gt_boxes[:, [0, 2]] *= w
            gt_boxes[:, [1, 3]] *= h
            gt_classes = list(gt[:, 4].astype(int))
            gt_masks = torch.Tensor(gt_masks).view(-1, h * w)

            if num_crowd > 0:
                split = lambda x: (x[-num_crowd:], x[:-num_crowd])
                crowd_boxes, gt_boxes = split(gt_boxes)
                crowd_masks, gt_masks = split(gt_masks)
                crowd_classes, gt_classes = split(gt_classes)

    with timer.env('Postprocess'):
        classes, scores, boxes, masks = postprocess(dets, w, h)

        if classes.size(0) == 0:
            return

        classes = list(classes.cpu().numpy().astype(int))
        scores = list(scores.cpu().numpy().astype(float))
        masks = masks.view(-1, h * w).cuda()
        boxes = boxes.cuda()

    if coco_json:
        with timer.env('JSON Output'):
            boxes = boxes.cpu().numpy()
            masks = masks.view(-1, h, w).cpu().numpy()

            for i in range(masks.shape[0]):
                # Make sure that the bounding box actually makes sense and a mask was produced
                if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                    make_json.add_bbox(image_id, classes[i], boxes[i, :], scores[i])
                    make_json.add_mask(image_id, classes[i], masks[i, :, :], scores[i])

            return

    with timer.env('Eval Setup'):
        num_pred = len(classes)
        num_gt = len(gt_classes)

        mask_iou_cache = mask_iou(masks, gt_masks)
        bbox_iou_cache = bbox_iou(boxes.float(), gt_boxes.float())

        if num_crowd > 0:
            crowd_mask_iou_cache = mask_iou(masks, crowd_masks, iscrowd=True)
            crowd_bbox_iou_cache = bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
        else:
            crowd_mask_iou_cache = None
            crowd_bbox_iou_cache = None

        iou_types = [('box', lambda i, j: bbox_iou_cache[i, j].item(), lambda i, j: crowd_bbox_iou_cache[i, j].item()),
                     ('mask', lambda i, j: mask_iou_cache[i, j].item(), lambda i, j: crowd_mask_iou_cache[i, j].item())]

    timer.start('Main loop')
    for _class in set(classes + gt_classes):
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])

        for iouIdx in range(len(iou_thresholds)):
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func, crowd_func in iou_types:
                gt_used = [False] * len(gt_classes)
                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in range(num_pred):
                    if classes[i] != _class:
                        continue

                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue

                        iou = iou_func(i, j)

                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j

                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(scores[i], True)
                    else:
                        # If the detection matches a crowd, we can just ignore it
                        matched_crowd = False

                        if num_crowd > 0:
                            for j in range(len(crowd_classes)):
                                if crowd_classes[j] != _class:
                                    continue

                                iou = crowd_func(i, j)

                                if iou > iou_threshold:
                                    matched_crowd = True
                                    break

                        # All this crowd code so that we can make sure that our eval code gives the
                        # same result as COCOEval. There aren't even that many crowd annotations to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            ap_obj.push(scores[i], False)
    timer.stop('Main loop')


class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score: float, is_true: bool):
        self.data_points.append((score, is_true))

    def add_gt_positives(self, num_positives: int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls = []
        num_true = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]:
                num_true += 1
            else:
                num_false += 1

            precision = num_true / (num_true + num_false)
            recall = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions) - 1, 0, -1):
            if precisions[i] > precisions[i - 1]:
                precisions[i - 1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101  # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)


def evaluate(net, dataset, during_training=False, trad_nms=False, max_img=-1, benchmark=False, coco_json=False):
    nms = NMS(top_k=200, conf_thresh=0.05, nms_thresh=0.5, trad_nms=trad_nms)

    frame_times = MovingAverage()
    dataset_size = len(dataset) if max_img < 0 else min(max_img, len(dataset))
    progress_bar = ProgressBar(30, dataset_size)

    print()
    if not benchmark:
        # For each class and iou, stores tuples (score, isPositive)
        # Index ap_data[type][iouIdx][classIdx]
        ap_data = {'box': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
                   'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds]}
        make_json = Make_json()
    else:
        timer.disable('Load Data')

    dataset_indices = list(range(len(dataset)))
    dataset_indices = dataset_indices[:dataset_size]

    try:
        for i, image_idx in enumerate(dataset_indices):
            timer.reset()

            with timer.env('Load Data'):
                img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)
                batch = Variable(img.unsqueeze(0))

                global cuda
                if cuda:
                    batch = batch.cuda()

            with timer.env('Network Extra'):
                predictions = net(batch)

            if benchmark:
                prep_benchmark(nms(predictions), h, w)
            else:
                prep_metrics(ap_data, nms(predictions), gt, gt_masks, h, w, num_crowd,
                             dataset.ids[image_idx], make_json, coco_json)

            # First couple of images take longer because we're constructing the graph.
            # Since that's technically initialization, don't include those in the FPS calculations.
            fps = 0
            if i > 1 and not during_training:
                frame_times.add(timer.total_time())
                fps = 1 / frame_times.get_avg()

            progress = (i + 1) / dataset_size * 100
            progress_bar.set_val(i + 1)
            print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps    ' % (
            repr(progress_bar), i + 1, dataset_size, progress, fps), end='')

        if benchmark:
            print('\n\n')
            print('Stats for the last frame:')
            timer.print_stats()
            avg_seconds = frame_times.get_avg()
            print('Average: %5.2f fps, %5.2f ms' % (1 / frame_times.get_avg(), 1000 * avg_seconds))

        else:
            if coco_json:
                make_json.dump()
                print('\nCompletely dumped results.')
                return

            table = calc_map(ap_data)
            print(table)
            return table

    except KeyboardInterrupt:
        print('Stopping...')


def calc_map(ap_data):
    print('\nCalculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]

    for _class in range(len(cfg.dataset.class_names)):
        for iou_idx in range(len(iou_thresholds)):
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0  # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold * 100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values()) - 1))

    row1 = list(all_maps['box'].keys())
    row1.insert(0, ' ')

    row2 = list(all_maps['box'].values())
    row2 = [round(aa, 2) for aa in row2]
    row2.insert(0, 'box')

    row3 = list(all_maps['mask'].values())
    row3 = [round(aa, 2) for aa in row3]
    row3.insert(0, 'mask')

    table = [row1, row2, row3]
    table = AsciiTable(table)
    return table.table


iou_thresholds = [x / 100 for x in range(50, 100, 5)]
cuda = torch.cuda.is_available()

if __name__ == '__main__':
    parse_args()

    if args.config is None:
        piece = args.trained_model.split('/')[1].split('_')
        name = f'{piece[0]}_{piece[1]}_config'
        print(f'Config not specified. Parsed \'{name}\' from the checkpoint name.\n')
        set_cfg(name)

    if not os.path.exists('results'):
        os.makedirs('results')

    with torch.no_grad():
        if cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info, augmentation=BaseTransform())

        print('Loading model...')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print('Done!')

        if cuda:
            net = net.cuda()

        evaluate(net, dataset, trad_nms=args.traditional_nms, max_img=args.max_images,
                 benchmark=args.benchmark, coco_json=args.output_coco_json)
