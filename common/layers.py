import cupy as np

from common.functions import (sigmoid, softmax, cross_entropy_error)

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class Affine:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)
        self.param = [self.W, self.b]

        self.grad, self.acquire_grad = None, True
        self.x, self.dW, self.db = None, None, None

    def forward(self, x):
        self.x = x

        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        self.grad = [self.dW, self.db]

        return dx


class ReLu:
    def __init__(self):
        self.mask = None
        self.acquire_grad = False

    def forward(self, x):
        self.mask = x > 0

        out = x[self.mask]

        return out

    def backward(self, dout):
        dout[~self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None
        self.acquire_grad = False

    def forward(self, x):
        out = sigmoid(x)
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.y, self.t = None, None
        self.acquire_grad = False

    def forward(self, x, t):
        self.y, self.t = softmax(x), t

        loss = cross_entropy_error(self.y, self.t)

        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        if self.y.size == self.t.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
