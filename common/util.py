import pickle
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


def progress_bar(now, total, message='', break_line=False, bar=False):
    if bar:
        count = int(((now + 1) / total) * 10)
        message += '[' + '-' * count + ' ' * (10 - count) + ']' + f' {count}/10'
    print(f'\r{message}', end='\n' if break_line else '')


def print_args(args):
    print('Hype: ', end='')
    for key, value in args.items():
        print(f'{key}:{value}', end='  ')
    print()


def save(file, model, optimizer):
    file = Path(file)
    with open(file / Path('model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(file / Path('optimizer.pkl'), 'wb') as f:
        pickle.dump(optimizer, f)


def load(file):
    file = Path(file)
    with open(file / Path('model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(file / Path('optimizer.pkl'), 'rb') as f:
        optimizer = pickle.load(f)

    return model, optimizer


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
