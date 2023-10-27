import cupy as np

from common.layers import (Affine, Convolution, Pooling, Flatten, ReLu, SoftmaxWithLoss)  # DO NOT MOVE

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class BaseModel:
    def __init__(self):
        self.layers, self.loss_layer, self.grads = None, None, None

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)

        return out

    def loss(self, y, t):
        loss = self.loss_layer.forward(y, t)

        return loss.item()

    def backward(self, dout=1):
        dx = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dx = layer.backward(dx)

        for layer in self.layers:
            if layer.acquire_grad:
                self.grads += layer.grad
                layer.zero_grad()

        return dx

    def predict(self, x):
        y = self.forward(x).argmax(axis=0) if x.ndim == 1 \
            else self.forward(x).argmax(axis=1)

        return y

    def accuracy(self, x, t):
        total_count = 1 if x.ndim == 1 else x.shape[0]
        y = self.predict(x)

        accu_count = np.sum(y == t)
        accuracy = accu_count / total_count

        return accuracy.item()


class Linear(BaseModel):
    def __init__(self, input_size, hidden_size_list, class_number, weight_init_std='xavier'):
        super().__init__()
        self.input_size, self.hidden_size_list, self.class_number = input_size, hidden_size_list, class_number
        self.layers, self.params, self.grads = [], [], []

        size_list = [input_size] + hidden_size_list + [class_number]
        for input_size, output_size in zip(size_list, size_list[1:]):
            self.layers.append(Affine(input_size, output_size, weight_init_std=weight_init_std))
            if output_size != size_list[-1]:
                self.layers.append(ReLu())
        self.loss_layer = SoftmaxWithLoss()

        for layer in self.layers:
            if layer.acquire_grad:
                self.params += layer.param


class Convolutional(BaseModel):
    def __init__(self, layer_params):
        super().__init__()
        self.layers, self.params, self.grads = [], [], []

        for layer_param in layer_params:
            self.layers.append(eval(layer_param['layer'])(*layer_param['param']))
        self.loss_layer = SoftmaxWithLoss()

        for layer in self.layers:
            if layer.acquire_grad:
                self.params += layer.param
