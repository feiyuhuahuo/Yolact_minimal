import re
import torch
import time
import argparse
import torch.utils.data as data
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch.backends.cudnn as cudnn

from utils.coco import COCODetection, val_collate
from modules.yolact import Yolact
from utils import timer
from utils.output_utils import after_nms, nms
from utils.common_utils import ProgressBar, MakeJson, APDataObject, prep_metrics, calc_map
from config import get_config

parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
parser.add_argument('--img_size', type=int, default=544, help='The image size for validation.')
parser.add_argument('--weight', type=str, default='weights/res101_coco_800000.pth', help='The validation model.')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--val_num', default=-1, type=int, help='The number of images for test, set to -1 for all.')
parser.add_argument('--coco_api', action='store_true', help='Whether to use cocoapi to evaluate results.')

iou_thres = [x / 100 for x in range(50, 100, 5)]
make_json = MakeJson()


def evaluate(net, cfg, step=None):
    dataset = COCODetection(cfg, mode='val')
    data_loader = data.DataLoader(dataset, 1, num_workers=4, shuffle=False, pin_memory=True, collate_fn=val_collate)
    ds = len(data_loader)
    progress_bar = ProgressBar(40, ds)
    timer.reset()

    ap_data = {'box': [[APDataObject() for _ in cfg.class_names] for _ in iou_thres],
               'mask': [[APDataObject() for _ in cfg.class_names] for _ in iou_thres]}

    for i, (img, gt, gt_masks, img_h, img_w) in enumerate(data_loader):
        if i == 1:
            timer.start()

        if cfg.cuda:
            img, gt, gt_masks = img.cuda(), gt.cuda(), gt_masks.cuda()

        with torch.no_grad(), timer.counter('forward'):
            class_p, box_p, coef_p, proto_p = net(img)

        with timer.counter('nms'):
            ids_p, class_p, box_p, coef_p, proto_p = nms(class_p, box_p, coef_p, proto_p, net.anchors, cfg)

        with timer.counter('after_nms'):
            ids_p, class_p, boxes_p, masks_p = after_nms(ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w)
            if ids_p is None:
                continue

        with timer.counter('metric'):
            ids_p = list(ids_p.cpu().numpy().astype(int))
            class_p = list(class_p.cpu().numpy().astype(float))

            if cfg.coco_api:
                boxes_p = boxes_p.cpu().numpy()
                masks_p = masks_p.cpu().numpy()

                for j in range(masks_p.shape[0]):
                    if (boxes_p[j, 3] - boxes_p[j, 1]) * (boxes_p[j, 2] - boxes_p[j, 0]) > 0:
                        make_json.add_bbox(dataset.ids[i], ids_p[j], boxes_p[j, :], class_p[j])
                        make_json.add_mask(dataset.ids[i], ids_p[j], masks_p[j, :, :], class_p[j])
            else:
                prep_metrics(ap_data, ids_p, class_p, boxes_p, masks_p, gt, gt_masks, img_h, img_w, iou_thres)

        aa = time.perf_counter()
        if i > 0:
            batch_time = aa - temp
            timer.add_batch_time(batch_time)
        temp = aa

        if i > 0:
            t_t, t_d, t_f, t_nms, t_an, t_me = timer.get_times(['batch', 'data', 'forward',
                                                                'nms', 'after_nms', 'metric'])
            fps, t_fps = 1 / (t_d + t_f + t_nms + t_an), 1 / t_t
            bar_str = progress_bar.get_bar(i + 1)
            print(f'\rTesting: {bar_str} {i + 1}/{ds}, fps: {fps:.2f} | total fps: {t_fps:.2f} | '
                  f't_t: {t_t:.3f} | t_d: {t_d:.3f} | t_f: {t_f:.3f} | t_nms: {t_nms:.3f} | '
                  f't_after_nms: {t_an:.3f} | t_metric: {t_me:.3f}', end='')

    if cfg.coco_api:
        make_json.dump()
        print(f'\nJson files dumped, saved in: \'results/\', start evaluating.')

        gt_annotations = COCO(cfg.val_ann)
        bbox_dets = gt_annotations.loadRes(f'results/bbox_detections.json')
        mask_dets = gt_annotations.loadRes(f'results/mask_detections.json')

        print('\nEvaluating BBoxes:')
        bbox_eval = COCOeval(gt_annotations, bbox_dets, 'bbox')
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()

        print('\nEvaluating Masks:')
        bbox_eval = COCOeval(gt_annotations, mask_dets, 'segm')
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()
    else:
        table, box_row, mask_row = calc_map(ap_data, iou_thres, len(cfg.class_names), step=step)
        print(table)
        return table, box_row, mask_row


if __name__ == '__main__':
    args = parser.parse_args()
    args.cfg = re.findall(r'res.+_[a-z]+', args.weight)[0]
    cfg = get_config(args, mode='val')

    net = Yolact(cfg)
    net.load_weights(cfg.weight, cfg.cuda)
    net.eval()

    if cfg.cuda:
        cudnn.benchmark = True
        cudnn.fastest = True
        net = net.cuda()

    evaluate(net, cfg)
