#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import glob
import torch
import os


class ProgressBar:
    def __init__(self, length, max_val):
        self.max_val = max_val
        self.length = length
        self.cur_val = 0

        self.cur_num_bars = -1
        self.update_str()

    def update_str(self):
        num_bars = int(self.length * (self.cur_val / self.max_val))

        if num_bars != self.cur_num_bars:
            self.cur_num_bars = num_bars
            self.string = '█' * num_bars + '░' * (self.length - num_bars)

    def get_bar(self, new_val):
        self.cur_val = new_val

        if self.cur_val > self.max_val:
            self.cur_val = self.max_val
        self.update_str()
        return self.string


def save_best(net, mask_map, cfg_name, step):
    weight = glob.glob('weights/best*')
    best_mask_map = float(weight[0].split('/')[-1].split('_')[1]) if weight else 0.

    if mask_map >= best_mask_map:
        if weight:
            os.remove(weight[0])  # remove the last best model

        print(f'\nSaving the best model as \'best_{mask_map}_{cfg_name}_{step}.pth\'.\n')
        torch.save(net.state_dict(), f'weights/best_{mask_map}_{cfg_name}_{step}.pth')


def save_latest(net, cfg_name, step):
    weight = glob.glob('weights/latest*')
    if weight:
        os.remove(weight[0])

    torch.save(net.state_dict(), f'weights/latest_{cfg_name}_{step}.pth')
