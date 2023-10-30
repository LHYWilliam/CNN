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


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None)
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--weight-init', type=str, default='he')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--noplot', action='store_true')
    parser.add_argument('--early-break', action='store_true')
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--train-show-per-iter', '--train-show', type=int, default=1)
    parser.add_argument('--test-show-per-iter', '--test-show', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


def print_args(args):
    print('\narguments: ', end='')
    for key, value in args.items():
        print(f'{key}:{value}', end='  ', flush=True)


def print_cfg(layers):
    print("\n\nnumber    layer               param")
    for number, layer_param in enumerate(layers):
        layer, param = layer_param.values()
        print(f'{number:<10}{layer:20}{param}')


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

        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                break
        path = Path(p)
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
