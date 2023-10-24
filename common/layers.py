import cupy as np

from common.functions import (softmax, cross_entropy_error)

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class Affine:
    def __init__(self, input_size, output_size, basis=True):
        self.W = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size) if basis else np.zeros(output_size)
        self.param = np.array([*self.W, self.b])
        self.grad = np.zeros_like(self.param)
        self.acquire_grad = True

    def forward(self, x):
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)

        return dx


class ReLu:
    def __init__(self):
        self.mask = None
        self.acquire_grad = False

    def forward(self, x):
        self.mask = x <= 0

        out = x.copy()
        out = out[self.mask]

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None
        self.acquire_grad = False

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss, self.y, self.t = None, None, None
        self.acquire_grad = False

    def forward(self, x, t):
        self.y, self.t = softmax(x), t

        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) / batch_size

        return dx
