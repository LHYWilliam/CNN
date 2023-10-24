import cupy as np

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for index in range(len(params)):
            params[index] -= self.lr * grads[index]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr, self.momentum = lr, momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = []

            for param in params:
                self.v.append(np.zeros_like(param))

        for index in range(len(params)):
            self.v[index] = self.momentum * self.v[index] - self.lr * grads[index]

            params[index] += self.v[index]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads, basis=1e-7):
        if self.h is None:
            self.h = []

            for param in params:
                self.h.append(np.zeros_like(param))

        for index in range(len(params)):
            self.h[index] += grads[index] * grads[index]

            params[index] -= self.lr * grads[index] / (np.sqrt(self.h[index]) + basis)


class Adam:
    def __init__(self, model, lr=0.01, beta1=0.9, beta2=0.999):
        self.params, self.grads = model.params, model.grads
        self.lr, self.beta1, self.beta2 = lr, beta1, beta2
        self.iter, self.m, self.v = 0, None, None

    def update(self, basis=1e-7):
        if self.m is None:
            self.m, self.v = [], []

            for param in self.params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for index in range(len(self.params)):
            self.m[index] += (1 - self.beta1) * (self.grads[index] - self.m[index])
            self.v[index] += (1 - self.beta2) * (self.grads[index] ** 2 - self.v[index])

            self.params[index] -= lr_t * self.m[index] / (np.sqrt(self.v[index]) + basis)
