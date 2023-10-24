import time

import numpy
import cupy as np
from matplotlib import pyplot as plt

from common.util import (progress_bar)

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class Trainer:
    def __init__(self, model, optimizer, val_per_iter=16):
        self.model, self.optimizer = model, optimizer
        self.val_per_iter = val_per_iter
        self.loss_list = []

    def train(self, x, t, goal_epoch=16, batch_size=128):
        total_size = x.shape[0]
        goal_iter = total_size // batch_size

        total_loss, loss_count = 0, 0
        for epoch in range(goal_epoch):
            start_time = time.time()
            index = numpy.random.permutation(numpy.arange(total_size))
            x, t = x[index], t[index]

            for iter in range(goal_iter):
                x_batch = x[iter * batch_size:(iter + 1) * batch_size]
                t_batch = t[iter * batch_size:(iter + 1) * batch_size]

                y = self.model.forward(x_batch)

                loss_count += 1
                total_loss += self.model.loss(y, t_batch)

                self.model.backward()
                self.optimizer.update()
                self.optimizer.zero_grad()

                if self.val_per_iter and ((iter + 1) % self.val_per_iter == 0 or iter + 1 == goal_iter):
                    average_loss = total_loss / loss_count
                    self.loss_list.append(float(average_loss))
                    message = f'| epoch {epoch + 1:{len(str(goal_epoch))}} ' \
                              f'| iter {iter + 1:{len(str(goal_iter))}}/{goal_iter} ' \
                              f'| loss {average_loss:.4f} ' \
                              f'| time {time.time() - start_time:.2f}s'
                    progress_bar(iter, goal_iter, message=message,
                                 break_line=(iter + 1 == goal_iter))

    def plot(self):
        x = numpy.arange(len(self.loss_list))

        plt.plot(x, self.loss_list, label='train')
        plt.xlabel(f'iterations (x{self.val_per_iter})')
        plt.ylabel('loss')

        plt.show()
