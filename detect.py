#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from modules.build_yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils.functions import MovingAverage, ProgressBar
from utils import timer
from data.config import set_cfg
from utils.output_utils import NMS, after_nms, draw_img
import torch
import torch.backends.cudnn as cudnn
import argparse
import glob
import cv2
import os

parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
parser.add_argument('--config', default=None, help='The config object of the model.')
parser.add_argument('--trained_model', default='weights/yolact_base_54_800000.pth', type=str)
parser.add_argument('--visual_top_k', default=100, type=int, help='Further restrict the number of predictions to parse')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--hide_mask', default=False, action='store_true', help='Whether to display masks')
parser.add_argument('--hide_bbox', default=False, action='store_true', help='Whether to display bboxes')
parser.add_argument('--hide_score', default=False, action='store_true', help='Whether to display scores')
parser.add_argument('--show_lincomb', default=False, action='store_true',
                    help='Whether to show the generating process of masks.')
parser.add_argument('--no_crop', default=False, action='store_true',
                    help='Do not crop output masks with the predicted bounding box.')
parser.add_argument('--image', default=None, type=str, help='The folder of images for detecting.')
parser.add_argument('--video', default=None, type=str,
                    help='A path to a video to evaluate on. Passing a number means using the related webcam.')
parser.add_argument('--visual_thre', default=0.3, type=float,
                    help='Detections with a score under this threshold will be removed.')

args = parser.parse_args()
if args.config is None:
    piece = args.trained_model.split('/')[1].split('_')
    name = f'{piece[0]}_{piece[1]}_config'
    print(f'\nConfig not specified. Parsed \'{name}\' from the checkpoint name.\n')
    set_cfg(name)

img_path = 'results/images'
video_path = 'results/videos'
if not os.path.exists(img_path):
    os.mkdir(img_path)
if not os.path.exists(video_path):
    os.mkdir(video_path)

with torch.no_grad():
    cuda = torch.cuda.is_available()
    if cuda:
        cudnn.benchmark = True
        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    print('Loading model...')
    net = Yolact()
    net.load_weights(args.trained_model)
    net.eval()

    if cuda:
        net = net.cuda()

    # detect images
    if args.image is not None:
        images = glob.glob(args.image + '/*.jpg')
        num = len(images)

        for i, one_img in enumerate(images):
            img_name = one_img.split('/')[-1]
            img_origin = torch.from_numpy(cv2.imread(one_img)).cuda().float()
            img_h, img_w = img_origin.shape[0], img_origin.shape[1]
            img_trans = FastBaseTransform()(img_origin.unsqueeze(0))
            net_outs = net(img_trans)
            nms_outs = NMS(net_outs, args.traditional_nms)

            show_lincomb = bool(args.show_lincomb and args.image_path)
            with timer.env('after nms'):
                results = after_nms(nms_outs, img_h, img_w, show_lincomb=show_lincomb, crop_masks=not args.no_crop,
                                    visual_thre=args.visual_thre, img_name=img_name)

                torch.cuda.synchronize()

            img_numpy = draw_img(results, img_origin, args)

            cv2.imwrite(f'{img_path}/{img_name}', img_numpy)
            print(f'{i + 1}/{num}', end='\r')

        print('\nDone.')

    # detect videos
    elif args.video is not None:
        vid = cv2.VideoCapture(args.video)

        target_fps = round(vid.get(cv2.CAP_PROP_FPS))
        frame_width = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        name = args.video.split('/')[-1]
        out = cv2.VideoWriter(f'{video_path}/{name}', cv2.VideoWriter_fourcc(*"mp4v"), target_fps,
                              (frame_width, frame_height))

        transform = FastBaseTransform()
        frame_times = MovingAverage()
        progress_bar = ProgressBar(40, num_frames)

        for i in range(num_frames):
            timer.reset()
            with timer.env('Detecting video'):
                frame_origin = torch.from_numpy(vid.read()[1]).cuda().float()
                img_h, img_w = frame_origin.shape[0], frame_origin.shape[1]
                frame_trans = transform(frame_origin.unsqueeze(0))
                net_outs = net(frame_trans)
                nms_outs = NMS(net_outs, args.traditional_nms)
                results = after_nms(nms_outs, img_h, img_w, crop_masks=not args.no_crop,
                                    visual_thre=args.visual_thre)
                torch.cuda.synchronize()

                frame_numpy = draw_img(results, frame_origin, args, class_color=True)
                out.write(frame_numpy)

            if i > 1:
                frame_times.add(timer.total_time())
                fps = 1 / frame_times.get_avg()
                progress = (i + 1) / num_frames * 100
                progress_bar.set_val(i + 1)

                print('\rProcessing Frames  %s %d / %d (%.2f%%) %.2f fps' % (
                    repr(progress_bar), i + 1, num_frames, progress, fps), end='')

        print(f'\nDone, saved in: results/videos/{name}')

        vid.release()
        out.release()
