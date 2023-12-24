import neolearn.config

if neolearn.config.Config.GPU:
    import cupy as np

    np.cuda.set_allocator(np.cuda.MemoryPool().malloc)
else:
    import numpy as np
