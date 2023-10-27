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


def progress_bar(now, total, message='', bar=False, break_line=False, basis=0.01):
    count = int((now / total + basis) * 10)
    if bar:
        message += '[' + '-' * count + ' ' * (10 - count) + ']' + f' {count}/10'
    print(f'\r{message}', end='\n' if break_line else '')


def print_args(args):
    print('Hype: ', end='')
    for key, value in args.items():
        print(f'{key}:{value}', end='  ')
    print()


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
