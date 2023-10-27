import time

import cupy as np

from common.util import (plots, progress_bar)

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class Trainer:
    def __init__(self, model, optimizer, train_show_per_iter=32, test_show_per_iter=1):
        self.model, self.optimizer = model, optimizer

        self.goal_epoch, self.train_iters, self.test_iters = None, None, None
        self.train_show_per_iter, self.test_show_per_iter = train_show_per_iter, test_show_per_iter

        self.loss_list, self.train_accuracy_list, self.test_accuracy_list = [], [], []

    def train(self, train_loader, test_loader, goal_epoch=16, batch_size=128, plot=True):
        self.goal_epoch, self.train_iters, self.test_iters = goal_epoch, len(train_loader), len(test_loader)

        for epoch in range(goal_epoch):
            total_loss, train_accu_count, test_accuracy = 0, 0, 0.

            start_time = time.time()
            for iter, (x_batch, t_batch) in enumerate(train_loader):

                y = self.model.forward(x_batch)

                total_loss += self.model.loss(y, t_batch)
                train_accu_count += np.sum(y.argmax(axis=1) == t_batch).item()

                self.model.backward()

                self.optimizer.update()
                self.optimizer.zero_grad()

                if self.train_show_per_iter and (
                        (iter + 1) % self.train_show_per_iter == 0 or iter + 1 == self.train_iters):
                    self._train_show(epoch, iter, batch_size, total_loss, train_accu_count, start_time)
                    total_loss, train_accu_count = 0, 0

            start_time = time.time()
            for iter, (x_batch, t_batch) in enumerate(test_loader):
                test_accuracy += self.model.val(x_batch, t_batch)

                if self.test_show_per_iter and (
                        (iter + 1) % self.test_show_per_iter == 0 or iter + 1 == self.test_iters):
                    self._test_show(iter, test_accuracy / self.test_show_per_iter, start_time)
                    test_accuracy = 0

        if plot:
            plots([self.loss_list], ['train loss'], [f'iter * {self.train_show_per_iter}'], ['loss'])
            plots([self.train_accuracy_list, self.test_accuracy_list], ['train accuracy', 'test accuracy'],
                  f'iter * {self.train_show_per_iter}', 'accuracy')

    def _train_show(self, epoch, iter, batch_size, total_loss, train_accu_count, start_time):
        interval_iter = self.train_show_per_iter if (iter + 1) % self.train_show_per_iter == 0 \
            else self.train_iters % self.train_show_per_iter

        average_loss = total_loss / interval_iter
        self.loss_list.append(average_loss)

        train_accuracy = train_accu_count / (batch_size * interval_iter)
        if iter + 1 == self.train_iters:
            self.train_accuracy_list.append(train_accuracy)

        message = f'| epoch {epoch + 1:{len(str(self.goal_epoch))}} ' \
                  f'| train ' \
                  f'| iter {iter + 1:{len(str(self.train_iters))}}/{self.train_iters} ' \
                  f'| loss {average_loss:.4f} ' \
                  f'| accuracy {train_accuracy:.4f} ' \
                  f'| time {time.time() - start_time:.2f}s '

        progress_bar(iter, self.train_iters, message=message, break_line=(iter + 1 == self.train_iters))

    def _test_show(self, iter, test_accuracy, start_time):
        epoch_blank = ' ' * len('| epoch  ' + str(self.goal_epoch))
        iter_blank = ' ' * (len(str(self.train_iters)) - len(str(self.test_iters))) * 2
        loss_blank = ' ' * (len(f'| loss  ') + 6)

        if iter + 1 == self.test_iters:
            self.test_accuracy_list.append(test_accuracy)

        message = f'{epoch_blank}' \
                  f'| test  ' \
                  f'| iter {iter + 1:{len(str(self.test_iters))}}/{self.test_iters} ' \
                  f'{iter_blank}' \
                  f'{loss_blank}' \
                  f'| accuracy {test_accuracy:.4f} ' \
                  f'| time {time.time() - start_time:.2f}s '

        progress_bar(iter, self.test_iters, message=message, break_line=(iter + 1 == self.test_iters))
