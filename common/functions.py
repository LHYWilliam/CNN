import cupy as np

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def softmax(x):
    y = x.copy()
    if y.ndim == 1:
        y = y.reshape(1, -1)
    y -= np.max(y, axis=1, keepdims=True)
    y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
    return y


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)

    if y.size == t.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
