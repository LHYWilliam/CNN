import abc
import cupy as np

from neolearn.layers import (Linear, Convolution, Pooling, Flatten, ReLu, BatchNormalization, Dropout)  # DO NOT MOVE

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class BaseModel(abc.ABC):
    def __init__(self):
        self.layers, self.grads = None, None

    def __call__(self, x):
        y = self.forward(x)

        return y

    def forward(self, x, train=True):
        out = x
        for layer in self.layers:
            out = layer.forward(out, train)

        return out

    def backward(self, dout=1):
        dx = dout
        for layer in reversed(self.layers):
            dx = layer.backward(dx)

        for layer in self.layers:
            if layer.acquire_grad:
                self.grads += layer.grad
                layer.zero_grad()

        return dx


class Model(BaseModel):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers, self.params, self.grads = [], [], []

        for layer_param in cfg:
            self.layers.append(eval(layer_param[0])(*layer_param[1]))

        for layer in self.layers:
            if layer.acquire_grad:
                self.params += layer.param

    def load(self, params):
        for self_param, param in zip(self.params, params):
            self_param[...] = param


class Sequential(BaseModel):
    def __init__(self, *args):
        super().__init__()
        self.layers, self.params, self.grads = args, [], []

        for layer in self.layers:
            if layer.acquire_grad:
                self.params += layer.param
