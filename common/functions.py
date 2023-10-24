import cupy as np

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    if x.ndim == 2:
        x -= np.max(x, axis=1, keepdims=True)
        x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    elif x.ndim == 1:
        x -= np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)

    if y.size == t.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
