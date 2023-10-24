import cupy as np

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class SGD:
    def __init__(self, model, lr=0.01):
        self.params, self.grads = model.params, model.grads
        self.lr = lr

    def update(self):
        for index in range(len(self.params)):
            self.params[index] -= self.lr * self.grads[index]

    def zero_grad(self):
        self.grads.clear()


class Momentum:
    def __init__(self, model, lr=0.01, momentum=0.9):
        self.params, self.grads = model.params, model.grads
        self.lr, self.momentum = lr, momentum
        self.v = None

    def update(self):
        if self.v is None:
            self.v = []

            for param in self.params:
                self.v.append(np.zeros_like(param))

        for index in range(len(self.params)):
            self.v[index] = self.momentum * self.v[index] - self.lr * self.grads[index]

            self.params[index] += self.v[index]

    def zero_grad(self):
        self.grads.clear()


class AdaGrad:
    def __init__(self, model, lr=0.01):
        self.params, self.grads = model.params, model.grads
        self.lr = lr
        self.h = None

    def update(self, basis=1e-7):
        if self.h is None:
            self.h = []

            for param in self.params:
                self.h.append(np.zeros_like(param))

        for index in range(len(self.params)):
            self.h[index] += self.grads[index] * self.grads[index]

            self.params[index] -= self.lr * self.grads[index] / (np.sqrt(self.h[index]) + basis)

    def zero_grad(self):
        self.grads.clear()


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

    def zero_grad(self):
        self.grads.clear()
