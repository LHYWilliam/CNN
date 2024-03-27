from neolearn.np import *


class Calculate:
    def __init__(self, train_iters, test_iters):
        self.train_iters, self.test_iters = train_iters, test_iters
        self._loss, self.train_accuracy, self.test_accuracy = [], [], []

    def train(self, loss, accuracy):
        self._loss.append(loss)
        self.train_accuracy.append(accuracy)

    def test(self, accuracy):
        self.test_accuracy.append(accuracy)

    @property
    def loss(self):
        return self._loss

    @property
    def average_loss(self):
        return np.average(self._loss[-self.train_iters // 10 :])

    @property
    def train_average_accuracy(self):
        return np.average(self.train_accuracy[-self.train_iters // 10 :])

    @property
    def test_average_accuracy(self):
        return np.average(self.test_accuracy[-self.test_iters :])

    @property
    def test_best_accuracy(self):
        return max(self.test_accuracy[: -1 : self.test_iters])

    @property
    def test_last_accuracy(self):
        return self.test_accuracy[-1]

    @property
    def train_epochs_accuracy(self):
        return self.train_accuracy[self.train_iters - 1 :: self.train_iters]

    @property
    def test_epochs_accuracy(self):
        return self.test_accuracy[self.test_iters - 1 :: self.test_iters]
