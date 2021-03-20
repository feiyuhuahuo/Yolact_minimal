#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import onnxruntime as ort
import argparse
import re
import pdb

from config import get_config
from modules.yolact import Yolact

parser = argparse.ArgumentParser(description='YOLACT Detection.')
parser.add_argument('--weight', default='weights/best_30.5_res101_coco_392000.pth', type=str)
parser.add_argument('--opset', type=int, default=12, help='The opset version for transporting to ONNX.')
parser.add_argument('--img_size', type=int, default=550, help='The image size for validation.')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')

args = parser.parse_args()
args.cfg = re.findall(r'res.+_[a-z]+', args.weight)[0]
cfg = get_config(args, mode='detect')

net = Yolact(cfg)
net.load_weights(cfg.weight, cfg.cuda)
net.eval().cuda()
print(f'Model loaded with {cfg.weight}.\n')

img_tensor = torch.randn((1, 3, cfg.img_size, cfg.img_size), device='cuda')
torch.onnx.export(net, img_tensor, f'onnx_files/{args.cfg}.onnx', verbose=False,
                  opset_version=args.opset, enable_onnx_checker=True)

sess = ort.InferenceSession(f'onnx_files/{args.cfg}.onnx')
input_name = sess.get_inputs()[0].name
img_numpy = img_tensor.cpu().numpy()

onnx_out = sess.run(None, {input_name: img_numpy})
torch_out = net(img_tensor)
torch_out = [aa.detach().cpu().numpy() for aa in torch_out]

for i, (aa, bb) in enumerate(zip(torch_out, onnx_out)):
    diff = (aa[0] - bb[0]).sum()
    if -1 < diff < 1:
        print(f'out: {i}, diff: {diff}')
    else:
        print(f'Error, diff is too large for out: {i}, export failed.')
        exit()

print(f'\nExported as `onnx_files/{args.cfg}.onnx`.')
