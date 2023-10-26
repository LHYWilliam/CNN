import time

import numpy
import cupy as np
from matplotlib import pyplot as plt

from common.util import (progress_bar)

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class Trainer:
    def __init__(self, model, optimizer, show_per_iter=32):
        self.model, self.optimizer = model, optimizer
        self.show_per_iter = show_per_iter
        self.loss_list = []

    def train(self, x, t, goal_epoch=16, batch_size=128):
        total_size = x.shape[0]
        goal_iter = total_size // batch_size

        for epoch in range(goal_epoch):
            start_time = time.time()
            index = numpy.random.permutation(numpy.arange(total_size))
            x, t = x[index], t[index]

            total_loss, train_accu_count = 0, 0
            for iter in range(goal_iter):
                x_batch = x[iter * batch_size:(iter + 1) * batch_size]
                t_batch = t[iter * batch_size:(iter + 1) * batch_size]

                y = self.model.forward(x_batch)

                total_loss += self.model.loss(y, t_batch)
                train_accu_count += np.sum(y.argmax(axis=1) == t_batch)

                self.model.backward()

                self.optimizer.update()
                self.optimizer.zero_grad()

                if self.show_per_iter and ((iter + 1) % self.show_per_iter == 0 or iter + 1 == goal_iter):
                    self._train_show(epoch, iter, goal_epoch, goal_iter, batch_size,
                                     total_loss, train_accu_count, start_time)
                    total_loss, train_accu_count = 0, 0

    def _train_show(self, epoch, iter, goal_epoch, goal_iter, batch_size, total_loss, train_accu_count, start_time):
        interval_iter = self.show_per_iter if (iter + 1) % self.show_per_iter == 0 \
            else goal_iter % self.show_per_iter

        average_loss = total_loss / interval_iter
        self.loss_list.append(float(average_loss))

        train_accuracy = train_accu_count / (batch_size * interval_iter)

        message = f'| epoch {epoch + 1:{len(str(goal_epoch))}} ' \
                  f'| iter {iter + 1:{len(str(goal_iter))}}/{goal_iter} ' \
                  f'| loss {average_loss:.4f} ' \
                  f'| train accuracy {train_accuracy:.4f}s ' \
                  f'| time {time.time() - start_time:.2f}s '

        progress_bar(iter, goal_iter, message=message, break_line=(iter + 1 == goal_iter))

    def plot(self):
        x = numpy.arange(len(self.loss_list))

        plt.plot(x, self.loss_list, label='train')
        plt.xlabel(f'iterations (x{self.show_per_iter})')
        plt.ylabel('loss')

        plt.show()
