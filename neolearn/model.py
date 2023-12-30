import abc

from neolearn.layers import (Linear, Convolution, Pooling, Flatten, ReLu, BatchNormalization, Dropout)  # DO NOT MOVE


class BaseModel(abc.ABC):
    def __init__(self):
        self.layers, self.params, self.grads = [], [], []

    def __call__(self, x, train=True):
        y = self.forward(x, train)

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

        self.layers = [eval(layer_param[0])(*layer_param[1]) for layer_param in cfg]
        self.params, self.grads = [], []

        for layer in self.layers:
            if layer.acquire_grad:
                self.params += layer.param

    def load(self, params):
        for i, param in zip(range(len(self.params)), params):
            self.params[i][...] = param


class Sequential(BaseModel):
    def __init__(self, *layers):
        super().__init__()
        self.layers, self.params, self.grads = layers, [], []

        for layer in self.layers:
            if layer.acquire_grad:
                self.params += layer.param
