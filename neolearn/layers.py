import cupy as np

from neolearn.util import (xavier, he, im2col, col2im)  # DO NOT MOVE
from neolearn.functions import (sigmoid, softmax, cross_entropy_error)

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class Linear:
    def __init__(self, input_size, output_size, weight_init='he'):
        self.W = eval(weight_init)(input_size) * np.random.randn(input_size, output_size)
        self.b = eval(weight_init)(input_size) * np.random.randn(output_size)
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
    def __init__(self, input_channel, out_channel, filter_size, stride=1, pad=0, weight_init='he'):
        self.W = (eval(weight_init)(input_channel * filter_size * filter_size) *
                  np.random.randn(out_channel, input_channel, filter_size, filter_size))
        self.b = np.zeros(out_channel)
        self.param = [self.W, self.b]

        self.stride, self.pad = stride, pad

        self.dW, self.db = None, None
        self.grad, self.acquire_grad = [], True

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
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        self.db, self.dW = np.sum(dout, axis=0), np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        self.grad = [self.dW, self.db]

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
        pool_size = self.h * self.w

        dout = dout.transpose(0, 2, 3, 1)
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


class BatchNormalization:
    def __init__(self, input_size, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = np.ones(input_size)
        self.beta = np.zeros(input_size)
        self.momentum = momentum
        self.running_mean, self.running_var = running_mean, running_var
        self.param = [self.gamma, self.beta]

        self.shape, self.batch_size = None, None
        self.xc, self.std = None, None

        self.grad, self.acquire_grad = [], True
        self.dgamma, self.dbeta = None, None

    def forward(self, x, train=True):
        self.shape, self.batch_size = x.shape, x.shape[0]

        if x.ndim != 2:
            x = x.reshape(self.batch_size, -1)

        out = self.__forward(x, train)
        out = out.reshape(*self.shape)

        return out

    def backward(self, dout):
        if dout.ndim != 2:
            dout = dout.reshape(self.batch_size, -1)

        dx = self.__backward(dout)
        dx = dx.reshape(*self.shape)

        self.grad = [self.dgamma, self.dbeta]

        return dx

    def zero_grad(self):
        self.grad.clear()

    def __forward(self, x, train):
        if self.running_mean is None:
            self.running_mean = np.zeros(x.shape[1])
            self.running_var = np.zeros(x.shape[1])

        if train:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.xc, self.xn, self.std = xc, xn, std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + 10e-7)

        out = self.gamma * xn + self.beta

        return out

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma, self.dbeta = dgamma, dbeta

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
        dx = dout * self.mask

        return dx
