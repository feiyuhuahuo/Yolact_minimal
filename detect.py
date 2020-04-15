#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.backends.cudnn as cudnn
import argparse
import glob
import cv2
import time

from modules.build_yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils.functions import MovingAverage, ProgressBar
from data.config import update_config
from utils.output_utils import NMS, after_nms, draw_img

parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
parser.add_argument('--trained_model', default='weights/yolact_base_54_800000.pth', type=str)
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--hide_mask', default=False, action='store_true', help='Whether to display masks')
parser.add_argument('--hide_bbox', default=False, action='store_true', help='Whether to display bboxes')
parser.add_argument('--hide_score', default=False, action='store_true', help='Whether to display scores')
parser.add_argument('--cutout', default=False, action='store_true', help='Whether to cut out each object')
parser.add_argument('--show_lincomb', default=False, action='store_true',
                    help='Whether to show the generating process of masks.')
parser.add_argument('--no_crop', default=False, action='store_true',
                    help='Do not crop output masks with the predicted bounding box.')
parser.add_argument('--image', default=None, type=str, help='The folder of images for detecting.')
parser.add_argument('--video', default=None, type=str,
                    help='The path of the video to evaluate. Pass a number to use the related webcam.')
parser.add_argument('--real_time', default=False, action='store_true', help='Show the detection results real-timely.')
parser.add_argument('--visual_thre', default=0.3, type=float,
                    help='Detections with a score under this threshold will be removed.')

args = parser.parse_args()
strs = args.trained_model.split('_')
config = f'{strs[-3]}_{strs[-2]}_config'

update_config(config)
print(f'\nUsing \'{config}\' according to the trained_model.\n')

with torch.no_grad():
    cuda = torch.cuda.is_available()
    if cuda:
        cudnn.benchmark = True
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    net = Yolact()
    net.load_weights('weights/' + args.trained_model, cuda)
    net.eval()
    print('Model loaded.\n')

    if cuda:
        net = net.cuda()

    # detect images
    if args.image is not None:
        images = glob.glob(args.image + '/*.jpg')

        for i, one_img in enumerate(images):
            img_name = one_img.split('/')[-1]
            img_origin = cv2.imread(one_img)
            img_tensor = torch.from_numpy(img_origin).float()
            if cuda:
                img_tensor = img_tensor.cuda()
            img_h, img_w = img_tensor.shape[0], img_tensor.shape[1]
            img_trans = FastBaseTransform()(img_tensor.unsqueeze(0))

            net_outs = net(img_trans)
            nms_outs = NMS(net_outs, args.traditional_nms)

            show_lincomb = bool(args.show_lincomb and args.image_path)
            results = after_nms(nms_outs, img_h, img_w, show_lincomb=show_lincomb, crop_masks=not args.no_crop,
                                visual_thre=args.visual_thre, img_name=img_name)

            img_numpy = draw_img(results, img_origin, img_name, args)
            cv2.imwrite(f'results/images/{img_name}', img_numpy)
            print(f'\r{i + 1}/{len(images)}', end='')

        print('\nDone.')

    # detect videos
    elif args.video is not None:
        vid = cv2.VideoCapture('videos/' + args.video)

        target_fps = round(vid.get(cv2.CAP_PROP_FPS))
        frame_width = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        name = args.video.split('/')[-1]
        video_writer = cv2.VideoWriter(f'results/videos/{name}', cv2.VideoWriter_fourcc(*"mp4v"), target_fps,
                                       (frame_width, frame_height))

        frame_times = MovingAverage()
        progress_bar = ProgressBar(40, num_frames)

        time_here = 0
        fps = 0
        for i in range(num_frames):
            frame_origin = torch.from_numpy(vid.read()[1]).float()
            if cuda:
                frame_origin = frame_origin.cuda()
            img_h, img_w = frame_origin.shape[0], frame_origin.shape[1]
            frame_trans = FastBaseTransform()(frame_origin.unsqueeze(0))
            net_outs = net(frame_trans)
            nms_outs = NMS(net_outs, args.traditional_nms)
            results = after_nms(nms_outs, img_h, img_w, crop_masks=not args.no_crop, visual_thre=args.visual_thre)

            if cuda:
                torch.cuda.synchronize()
            temp = time_here
            time_here = time.time()

            if i > 0:
                frame_times.add(time_here - temp)
                fps = 1 / frame_times.get_avg()

            frame_numpy = draw_img(results, frame_origin, args, fps=fps)

            if args.real_time:
                cv2.imshow('Detection', frame_numpy)
                cv2.waitKey(1)
            else:
                video_writer.write(frame_numpy)

            progress = (i + 1) / num_frames * 100
            progress_bar.set_val(i + 1)
            print(f'\rDetecting: {repr(progress_bar)} {i + 1} / {num_frames} ({progress:.2f}%) {fps:.2f} fps', end='')

        if not args.real_time:
            print(f'\n\nDone, saved in: results/videos/{name}')

        vid.release()
        video_writer.release()
