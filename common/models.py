import cupy as np

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)