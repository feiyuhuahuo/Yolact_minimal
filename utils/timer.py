import time
import torch
import numpy as np

times = {}
times.setdefault('batch', [])
times.setdefault('data', [])
mark = False  # Use for starting and stopping the timer
max_len = 100


def reset(length=100):
    global times, mark, max_len
    times = {}
    times.setdefault('batch', [])
    times.setdefault('data', [])
    mark = False
    max_len = length


def start():
    global mark, times
    mark = True

    for k, v in times.items():
        if len(v) != 0:
            print('Warning, time list is not empty when starting.')


def add_batch_time(batch_time):
    if mark:
        times['batch'].append(batch_time)

        inner_time = 0
        for k, v in times.items():
            if k not in ('batch', 'data'):
                inner_time += v[-1]

        times['data'].append(batch_time - inner_time)


def get_times(time_name):
    return_time = []
    for name in time_name:
        return_time.append(np.mean(times[name]))

    return return_time


class counter:
    def __init__(self, name):
        self.name = name
        self.times = times
        self.mark = mark
        self.max_len = max_len

        for v in times.values():
            if len(v) >= self.max_len:  # pop the first item if the list is full
                v.pop(0)

    def __enter__(self):
        if self.mark:
            torch.cuda.synchronize()
            self.times.setdefault(self.name, [])
            self.times[self.name].append(time.perf_counter())

    def __exit__(self, e, ev, t):
        if self.mark:
            torch.cuda.synchronize()
            self.times[self.name][-1] = time.perf_counter() - self.times[self.name][-1]

