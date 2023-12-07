import abc
import cupy as np

from neolearn.layers import (Affine, Convolution, Pooling, Flatten, ReLu, BatchNormalization, Dropout)  # DO NOT MOVE

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class BaseModel(abc.ABC):
    def __init__(self):
        pass

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

        return dx

    def predict(self, x):
        y = self.forward(x, train=False).argmax(axis=0) if x.ndim == 1 \
            else self.forward(x, train=False).argmax(axis=1)

        return y

    def accuracy(self, x, t):
        total_count = 1 if x.ndim == 1 else x.shape[0]
        y = self.predict(x)

        accu_count = np.sum(y == t)
        accuracy = accu_count / total_count

        return accuracy.item()


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