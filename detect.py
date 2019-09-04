#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from modules.build_yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils.functions import MovingAverage, ProgressBar
from utils import timer
from utils.functions import SavePath
from utils.output_utils import postprocess, NMS
from data.config import cfg, set_cfg, COLORS
import torch
import torch.backends.cudnn as cudnn
import argparse
import os
from collections import defaultdict
import glob
import cv2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model', default='weights/yolact_base_54_800000.pth', type=str)
    parser.add_argument('--top_k', default=100, type=int, help='Further restrict the number of predictions to parse')
    parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
    parser.add_argument('--hide_mask', default=False, action='store_true', help='Whether to display masks')
    parser.add_argument('--hide_bbox', default=False, action='store_true', help='Whether to display bboxes')
    parser.add_argument('--hide_score', default=False, action='store_true', help='Whether to display scores')
    parser.add_argument('--config', default=None, help='The config to use.')
    parser.add_argument('--show_lincomb', default=False, action='store_true', help='Whether to show that how masks are created.')
    parser.add_argument('--no_crop', default=False, action='store_true',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image_path', default=None, type=str, help='The folder of images for detecting.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--score_threshold', default=0.3, type=float,
                        help='Detections with a score under this threshold will be removed.')

    global args
    args = parser.parse_args(argv)


def prep_display(dets_out, img, class_color=False, mask_alpha=0.45, img_name=None):
    show_lincomb = bool(args.show_lincomb and args.image_path)

    img_gpu = img / 255.0
    h, w, _ = img.shape

    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, visualize_lincomb=show_lincomb, crop_masks=not args.no_crop,
                        score_threshold=args.score_threshold, img_name=img_name)
        torch.cuda.synchronize()

    with timer.env('Copy'):
        # Masks are drawn on the GPU, so don't copy
        masks = t[3][:args.top_k]
        classes, scores, boxes = [x[:args.top_k].cpu().numpy() for x in t[:3]]

    num_considered = min(args.top_k, classes.shape[0])
    for j in range(num_considered):
        if scores[j] < args.score_threshold:
            num_considered = j
            break

    if num_considered == 0:
        # No detections found so just output the original image
        return (img_gpu * 255).byte().cpu().numpy()

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            # The image might come in as RGB or BRG, depending
            color = (color[2], color[1], color[0])

            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    if not args.hide_mask:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_considered, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat(
            [get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_considered)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_considered > 1:
            inv_alph_cumul = inv_alph_masks[:(num_considered - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if not args.hide_bbox:
        for j in reversed(range(num_considered)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            _class = cfg.dataset.class_names[classes[j]]
            text_str = '%s: %.2f' % (_class, score) if not args.hide_score else _class
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 0.6
            thickness = 1

            text_w, text_h = cv2.getTextSize(text_str, font, scale, thickness)[0]

            cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 + text_h + 5), color, -1)
            cv2.putText(img_numpy, text_str, (x1, y1 + 15), font, scale, [255, 255, 255], thickness, cv2.LINE_AA)

    return img_numpy


def video(net: Yolact, in_path: str):
    vid = cv2.VideoCapture(in_path)

    target_fps = round(vid.get(cv2.CAP_PROP_FPS))
    frame_width = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    name = in_path.split('/')[-1]
    out = cv2.VideoWriter(f'results/videos/{name}', cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))

    transform = FastBaseTransform()
    frame_times = MovingAverage()
    progress_bar = ProgressBar(30, num_frames)

    try:
        for i in range(num_frames):
            timer.reset()
            with timer.env('Video'):
                frame = torch.from_numpy(vid.read()[1]).cuda().float()
                batch = transform(frame.unsqueeze(0))
                preds = net(batch)
                global nms
                processed = prep_display(nms(preds), frame, class_color=True)

                out.write(processed)

            if i > 1:
                frame_times.add(timer.total_time())
                fps = 1 / frame_times.get_avg()
                progress = (i + 1) / num_frames * 100
                progress_bar.set_val(i + 1)

                print('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps' % (
                    repr(progress_bar), i + 1, num_frames, progress, fps), end='')

        print(f'Done, saved in: results/videos/{name}')
    except KeyboardInterrupt:
        print('Stopping early.')

    vid.release()
    out.release()
    print()


if __name__ == '__main__':
    parse_args()
    color_cache = defaultdict(lambda: {})

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        config = model_path.model_name + '_config'
        print(f'Config not specified. Parsed \'{config}\' from the file name.\n')
        set_cfg(config)
        
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
        print(' Done.')

        if cuda:
            net = net.cuda()

        nms = NMS(top_k=200, conf_thresh=0.05, nms_thresh=0.5, trad_nms=args.traditional_nms)

        # detect images
        if args.image_path is not None:
            images = glob.glob(args.image_path + '/*')
            num = len(images)

            for i, one_img in enumerate(images):
                name = one_img.split('/')[-1]
                frame = torch.from_numpy(cv2.imread(one_img)).cuda().float()
                batch = FastBaseTransform()(frame.unsqueeze(0))
                preds = net(batch)
                img_numpy = prep_display(nms(preds), frame, img_name=name)
                cv2.imwrite(f'{save_path}/{name}', img_numpy)
                print(f'{i+1}/{num}', end='\r')

            print('\nDone.')

        # detect videos
        elif args.video is not None:
            video(net, args.video)
