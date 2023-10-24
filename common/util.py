import numpy
import cupy as np

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


def progress_bar(now, total, message='', basis=0.01):
    count = int((now / total + basis) * 10)
    print(f'\r{message} [' + '-' * count + ' ' * (10 - count) + ']' +
          f' {count}/10', end='')


def to_gpu(x):
    import cupy
    if type(x) is cupy.ndarray:
        return x
    return cupy.asarray(x)


def to_cpu(x):
    import numpy
    if type(x) is numpy.ndarray:
        return x
    return np.asnumpy(x)
