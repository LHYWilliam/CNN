from neolearn.np import *
from neolearn.functions import softmax, cross_entropy_error


class SoftmaxWithLoss:
    def __init__(self, model):
        self.model = model
        self.acquire_grad = False

        self.y, self.t = None, None

    def __call__(self, y, t, train=True):
        return self.forward(y, t, train=train)

    def forward(self, x, t, train=True):
        self.y, self.t = softmax(x), t

        loss = cross_entropy_error(self.y, self.t)

        return loss.item()

    def backward(self, dout=1):
        dx = self._loss_backward(dout)
        dx = self.model.backward(dx)

        return dx

    def _loss_backward(self, dout=1):
        batch_size = self.t.shape[0]

        if self.y.size == self.t.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx
