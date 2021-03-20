#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import argparse
import cv2
import time
import re
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from modules.yolact import Yolact
from config import get_config
from utils.coco import COCODetection, detect_collate
from utils import timer
from utils.output_utils import nms, after_nms, draw_img
from utils.common_utils import ProgressBar
from utils.augmentations import val_aug

parser = argparse.ArgumentParser(description='YOLACT Detection.')
parser.add_argument('--weight', default='weights/best_30.5_res101_coco_392000.pth', type=str)
parser.add_argument('--image', default=None, type=str, help='The folder of images for detecting.')
parser.add_argument('--video', default=None, type=str, help='The path of the video to evaluate.')
parser.add_argument('--img_size', type=int, default=550, help='The image size for validation.')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--hide_mask', default=False, action='store_true', help='Hide masks in results.')
parser.add_argument('--hide_bbox', default=False, action='store_true', help='Hide boxes in results.')
parser.add_argument('--hide_score', default=False, action='store_true', help='Hide scores in results.')
parser.add_argument('--cutout', default=False, action='store_true', help='Cut out each object and save.')
parser.add_argument('--save_lincomb', default=False, action='store_true', help='Show the generating process of masks.')
parser.add_argument('--no_crop', default=False, action='store_true',
                    help='Do not crop the output masks with the predicted bounding box.')
parser.add_argument('--real_time', default=False, action='store_true', help='Show the detection results real-timely.')
parser.add_argument('--visual_thre', default=0.3, type=float,
                    help='Detections with a score under this threshold will be removed.')

args = parser.parse_args()
args.cfg = re.findall(r'res.+_[a-z]+', args.weight)[0]
cfg = get_config(args, mode='detect')

net = Yolact(cfg)
net.load_weights(cfg.weight, cfg.cuda)
net.eval()
print(f'Model loaded with {cfg.weight}.\n')

if cfg.cuda:
    cudnn.benchmark = True
    cudnn.fastest = True
    net = net.cuda()

with torch.no_grad():
    # detect images
    if cfg.image is not None:
        dataset = COCODetection(cfg, mode='detect')
        data_loader = data.DataLoader(dataset, 1, num_workers=4, shuffle=False,
                                      pin_memory=True, collate_fn=detect_collate)
        ds = len(data_loader)
        assert ds > 0, 'No .jpg images found.'
        progress_bar = ProgressBar(40, ds)
        timer.reset()

        for i, (img, img_origin, img_name) in enumerate(data_loader):
            if i == 1:
                timer.start()

            if cfg.cuda:
                img = img.cuda()

            img_h, img_w = img_origin.shape[0:2]

            with timer.counter('forward'):
                class_p, box_p, coef_p, proto_p, anchors = net(img)

            with timer.counter('nms'):
                ids_p, class_p, box_p, coef_p, proto_p = nms(class_p, box_p, coef_p, proto_p, anchors, cfg)

            with timer.counter('after_nms'):
                ids_p, class_p, boxes_p, masks_p = after_nms(ids_p, class_p, box_p, coef_p,
                                                             proto_p, img_h, img_w, cfg, img_name=img_name)

            with timer.counter('save_img'):
                img_numpy = draw_img(ids_p, class_p, boxes_p, masks_p, img_origin, cfg, img_name=img_name)
                cv2.imwrite(f'results/images/{img_name}', img_numpy)

            aa = time.perf_counter()
            if i > 0:
                batch_time = aa - temp
                timer.add_batch_time(batch_time)
            temp = aa

            if i > 0:
                t_t, t_d, t_f, t_nms, t_an, t_si = timer.get_times(['batch', 'data', 'forward',
                                                                    'nms', 'after_nms', 'save_img'])
                fps, t_fps = 1 / (t_d + t_f + t_nms + t_an), 1 / t_t
                bar_str = progress_bar.get_bar(i + 1)
                print(f'\rTesting: {bar_str} {i + 1}/{ds}, fps: {fps:.2f} | total fps: {t_fps:.2f} | '
                      f't_t: {t_t:.3f} | t_d: {t_d:.3f} | t_f: {t_f:.3f} | t_nms: {t_nms:.3f} | '
                      f't_after_nms: {t_an:.3f} | t_save_img: {t_si:.3f}', end='')

        print('\nFinished, saved in: results/images.')

    # detect videos
    elif cfg.video is not None:
        vid = cv2.VideoCapture(cfg.video)

        target_fps = round(vid.get(cv2.CAP_PROP_FPS))
        frame_width = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        name = cfg.video.split('/')[-1]
        video_writer = cv2.VideoWriter(f'results/videos/{name}', cv2.VideoWriter_fourcc(*"mp4v"), target_fps,
                                       (frame_width, frame_height))

        progress_bar = ProgressBar(40, num_frames)
        timer.reset()
        t_fps = 0

        for i in range(num_frames):
            if i == 1:
                timer.start()

            frame_origin = vid.read()[1]
            img_h, img_w = frame_origin.shape[0:2]
            frame_trans = val_aug(frame_origin, cfg.img_size)

            frame_tensor = torch.tensor(frame_trans).float()
            if cfg.cuda:
                frame_tensor = frame_tensor.cuda()

            with timer.counter('forward'):
                class_p, box_p, coef_p, proto_p, anchors = net(frame_tensor.unsqueeze(0))

            with timer.counter('nms'):
                ids_p, class_p, box_p, coef_p, proto_p = nms(class_p, box_p, coef_p, proto_p, anchors, cfg)

            with timer.counter('after_nms'):
                ids_p, class_p, boxes_p, masks_p = after_nms(ids_p, class_p, box_p, coef_p,
                                                             proto_p, img_h, img_w, cfg)

            with timer.counter('save_img'):
                frame_numpy = draw_img(ids_p, class_p, boxes_p, masks_p, frame_origin, cfg, fps=t_fps)

            if cfg.real_time:
                cv2.imshow('Detection', frame_numpy)
                cv2.waitKey(1)
            else:
                video_writer.write(frame_numpy)

            aa = time.perf_counter()
            if i > 0:
                batch_time = aa - temp
                timer.add_batch_time(batch_time)
            temp = aa

            if i > 0:
                t_t, t_d, t_f, t_nms, t_an, t_si = timer.get_times(['batch', 'data', 'forward',
                                                                    'nms', 'after_nms', 'save_img'])
                fps, t_fps = 1 / (t_d + t_f + t_nms + t_an), 1 / t_t
                bar_str = progress_bar.get_bar(i + 1)
                print(f'\rDetecting: {bar_str} {i + 1}/{num_frames}, fps: {fps:.2f} | total fps: {t_fps:.2f} | '
                      f't_t: {t_t:.3f} | t_d: {t_d:.3f} | t_f: {t_f:.3f} | t_nms: {t_nms:.3f} | '
                      f't_after_nms: {t_an:.3f} | t_save_img: {t_si:.3f}', end='')

        if not cfg.real_time:
            print(f'\n\nFinished, saved in: results/videos/{name}')

        vid.release()
        video_writer.release()
