#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
import cv2
import time
import re
import math
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch.utils.data as data
import pdb

from config import get_config
from utils.coco import COCODetection, detect_onnx_collate
from utils import timer
from utils.output_utils import nms_numpy, after_nms_numpy, draw_img
from utils.common_utils import ProgressBar
from utils.box_utils import make_anchors
from utils.augmentations import val_aug

parser = argparse.ArgumentParser(description='YOLACT Detection with TensorRT.')
parser.add_argument('--weight', default='trt_files/res101_coco.trt', type=str)
parser.add_argument('--image', default=None, type=str, help='The folder of images for detecting.')
parser.add_argument('--video', default=None, type=str, help='The path of the video to evaluate.')
parser.add_argument('--img_size', type=int, default=544, help='The image size for validation.')
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
prefix = re.findall(r'best_\d+\.\d+_', args.weight)[0]
suffix = re.findall(r'_\d+\.pth', args.weight)[0]
args.cfg = args.weight.split(prefix)[-1].split(suffix)[0]
cfg = get_config(args, mode='detect')


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

anchors = []
fpn_fm_shape = [math.ceil(cfg.img_size / stride) for stride in (8, 16, 32, 64, 128)]
for i, size in enumerate(fpn_fm_shape):
    anchors += make_anchors(cfg, size, size, cfg.scales[i])

# prepare engine
with open(cfg.weight, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))

        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

# ------------------------------------------------------------------------------------------------------------
# Since also the inference procedure are done on GPU, so any other CUDA relevant operation should be excluded,
# e.g. CUDA operation in PyTorch, or some unexpected error may occur.
# ------------------------------------------------------------------------------------------------------------

# detect images
if cfg.image is not None:
    dataset = COCODetection(cfg, mode='detect')
    # Only num_workers=0 and pin_memory=True or num_workers>0 and pin_memory=False is OK, if use num_workers>0
    # and pin_memory=True, encounter error:
    # PyCUDA WARNING: a clean-up operation failed (dead context maybe?)
    # cuMemFreeHost failed: context is destroyed
    data_loader = data.DataLoader(dataset, 1, num_workers=4, shuffle=False,
                                  pin_memory=False, collate_fn=detect_onnx_collate)

    ds = len(data_loader)
    assert ds > 0, 'No .jpg images found.'
    progress_bar = ProgressBar(40, ds)
    timer.reset()

    for i, (img, img_origin, img_name) in enumerate(data_loader):
        if i == 1:
            timer.start()

        with timer.counter('forward', trt_mode=True), engine.create_execution_context() as context:
            assert img.shape == (1, 3, cfg.img_size, cfg.img_size), 'Img shape error.'
            inputs[0].host = img  # input dtype should be float32
            # Transfer input data to the GPU.
            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
            # Run inference.
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
            # Synchronize the stream
            stream.synchronize()
            # Return only the host outputs.
            results = [out.host for out in outputs]

            class_p = results[3].reshape(1, -1, cfg.num_classes)
            box_p = results[0].reshape(1, -1, 4)
            coef_p = results[1].reshape(1, -1, 32)
            proto_p = results[2].reshape(1, int(cfg.img_size / 4), int(cfg.img_size / 4), 32)

        with timer.counter('nms', trt_mode=True):
            ids_p, class_p, box_p, coef_p, proto_p = nms_numpy(class_p, box_p, coef_p, proto_p, anchors, cfg)

        with timer.counter('after_nms', trt_mode=True):
            img_h, img_w = img_origin.shape[0:2]
            ids_p, class_p, boxes_p, masks_p = after_nms_numpy(ids_p, class_p, box_p, coef_p,
                                                               proto_p, img_h, img_w, cfg)

        with timer.counter('save_img', trt_mode=True):
            img_numpy = draw_img(ids_p, class_p, boxes_p, masks_p, img_origin, cfg, img_name=img_name)
            cv2.imwrite(f'results/trt_images/{img_name}', img_numpy)

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

    print('\nFinished, saved in: results/trt_images.')

# detect videos
elif cfg.video is not None:
    vid = cv2.VideoCapture(cfg.video)

    target_fps = round(vid.get(cv2.CAP_PROP_FPS))
    frame_width = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    name = cfg.video.split('/')[-1]
    video_writer = cv2.VideoWriter(f'results/trt_videos/{name}', cv2.VideoWriter_fourcc(*"mp4v"), target_fps,
                                   (frame_width, frame_height))

    progress_bar = ProgressBar(40, num_frames)
    timer.reset()
    t_fps = 0

    for i in range(num_frames):
        if i == 1:
            timer.start()

        frame_origin = vid.read()[1]
        img_h, img_w = frame_origin.shape[0:2]
        frame_trans = val_aug(frame_origin, cfg.img_size)[None, :]

        with timer.counter('forward', trt_mode=True), engine.create_execution_context() as context:
            assert frame_trans.shape == (1, 3, cfg.img_size, cfg.img_size), 'Img shape error.'
            inputs[0].host = frame_trans  # input dtype should be float32
            # Transfer input data to the GPU.
            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
            # Run inference.
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
            # Synchronize the stream
            stream.synchronize()
            # Return only the host outputs.
            results = [out.host for out in outputs]

            class_p = results[3].reshape(1, -1, cfg.num_classes)
            box_p = results[0].reshape(1, -1, 4)
            coef_p = results[1].reshape(1, -1, 32)
            proto_p = results[2].reshape(1, int(cfg.img_size / 4), int(cfg.img_size / 4), 32)

        with timer.counter('nms', trt_mode=True):
            ids_p, class_p, box_p, coef_p, proto_p = nms_numpy(class_p, box_p, coef_p, proto_p, anchors, cfg)

        with timer.counter('after_nms', trt_mode=True):
            ids_p, class_p, boxes_p, masks_p = after_nms_numpy(ids_p, class_p, box_p, coef_p,
                                                               proto_p, img_h, img_w, cfg)

        with timer.counter('save_img', trt_mode=True):
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
        print(f'\n\nFinished, saved in: results/trt_videos/{name}')

    vid.release()
    video_writer.release()
