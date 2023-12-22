import os
import pickle
import argparse
from pathlib import Path

import numpy
import cupy
import cupy as np
from matplotlib import pyplot as plt

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


def xavier(n):
    return 1.0 / np.sqrt(n)


def he(n):
    return 2.0 / np.sqrt(n)


def plots(lists, labels, xlabel, ylabel):
    x = numpy.arange(1, len(lists[0]) + 1)
    for y, label in zip(lists, labels):
        plt.plot(x, y, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def progress_bar(now, total, message='', break_line=False, bar=True):
    if bar:
        count = int(((now + 1) / total) * 10)
        message += '     [' + '-' * count + ' ' * (10 - count) + ']' + f' {count}/10'
    print(f'\r{message}', end='\n' if break_line else '')


def save(file, checkpoint):
    with open(file, 'wb') as f:
        pickle.dump(checkpoint, f)


def load(file):
    with open(file, 'rb') as f:
        checkpoint = pickle.load(f)

    return checkpoint


def increment_path(path, sep='', mkdir=True, exist_ok=True):
    path = Path(path)
    if path.exists():
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        back = path

        for n in range(2, 9999):
            path = Path(f'{back}{sep}{n}{suffix}')
            if not path.exists():
                break

    if mkdir:
        path.mkdir(parents=True, exist_ok=exist_ok)

    return path


def im2col(input_data, h, w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - h) // stride + 1
    out_w = (W + 2 * pad - w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, h, w, out_h, out_w))

    for y in range(h):
        y_max = y + stride * out_h
        for x in range(w):
            x_max = x + stride * out_w

            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)

    return col


def col2im(col, input_shape, h, w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - h) // stride + 1
    out_w = (W + 2 * pad - w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, h, w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    for y in range(h):
        y_max = y + stride * out_h
        for x in range(w):
            x_max = x + stride * out_w

            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def print_args(args):
    print('\narguments: ', end='')
    for key, value in args.items():
        print(f'{key}:{value}', end='  ', flush=True)


def print_cfg(layer_param):
    print("\n\nnumber    layer               param")
    for number, layer_param in enumerate(layer_param):
        layer, param = layer_param
        print(f'{number:<10}{layer:20}{param}')


def to_gpu(*args):
    out = []

    for x in args:
        out.append(x if type(x) is cupy.ndarray else cupy.asarray(x))

    return out


def to_cpu(*args):
    out = []

    for x in args:
        out.append(x if type(x) is numpy.ndarray else cupy.asnumpy(x))

    return out
