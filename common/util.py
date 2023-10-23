import numpy
import cupy as np

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


def to_gpu(x):
    pass


def to_cpu(x):
    pass
