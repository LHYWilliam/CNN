import cupy as np

from common.util import (xavier, he, im2col, col2im)  # DO NOT MOVE
from common.functions import (sigmoid, softmax, cross_entropy_error)

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class Affine:
    def __init__(self, input_size, output_size, weight_init_std='xavier'):
        self.W = eval(weight_init_std)(input_size) * np.random.randn(input_size, output_size)
        self.b = eval(weight_init_std)(input_size) * np.random.randn(output_size)
        self.param = [self.W, self.b]

        self.dW, self.db = None, None
        self.grad, self.acquire_grad = [], True

        self.x = None

    def forward(self, x, train=True):
        self.x = x

        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        self.grad = [self.dW, self.db]

        return dx

    def zero_grad(self):
        self.grad.clear()


class Convolution:
    def __init__(self, filter_number, channel, filter_size, stride=1, pad=0, weight_init_std='xavier'):
        self.W = 0.01 * np.random.randn(filter_number, channel, filter_size, filter_size)
        self.b = np.zeros(filter_number)
        self.param = [self.W, self.b]

        self.dW, self.db = None, None
        self.grad, self.acquire_grad = [], True

        self.stride, self.pad = stride, pad
        self.x, self.col, self.col_W = None, None, None

    def forward(self, x, train=True):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x, self.col, self.col_W = x, col, col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db, self.dW = np.sum(dout, axis=0), np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        self.grad += [self.dW, self.db]

        return dx

    def zero_grad(self):
        self.grad.clear()


class Pooling:
    def __init__(self, h, w, stride=1, pad=0):
        self.h, self.w, self.stride, self.pad = h, w, stride, pad

        self.acquire_grad = False

        self.x, self.arg_max = None, None

    def forward(self, x, train=True):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.h) / self.stride)
        out_w = int(1 + (W - self.w) / self.stride)

        col = im2col(x, self.h, self.w, self.stride, self.pad)
        col = col.reshape(-1, self.h * self.w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x, self.arg_max = x, arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.h * self.w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.h, self.w, self.stride, self.pad)

        return dx


class Flatten:
    def __init__(self):
        self.acquire_grad = False

        self.x, self.shape = None, None

    def forward(self, x, train=True):
        self.shape = x.shape
        out = x.reshape(x.shape[0], -1)

        return out

    def backward(self, dout):
        dx = dout.reshape(*self.shape)

        return dx


class ReLu:
    def __init__(self):
        self.mask = None
        self.acquire_grad = False

    def forward(self, x, train=True):
        self.mask = x <= 0

        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None
        self.acquire_grad = False

    def forward(self, x, train=True):
        out = sigmoid(x)
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Dropout:
    def __init__(self, ratio=0.5):
        self.ratio, self.mask = ratio, None

        self.acquire_grad = False

    def forward(self, x, train=True):
        if train:
            self.mask = np.random.rand(*x.shape) > self.ratio
            return x * self.mask
        else:
            return x * (1.0 - self.ratio)

    def backward(self, dout):
        return dout * self.mask


class SoftmaxWithLoss:
    def __init__(self):
        self.y, self.t = None, None
        self.acquire_grad = False

    def forward(self, x, t, train=True):
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

# TODO: class BatchNormalization
