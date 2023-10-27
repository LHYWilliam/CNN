import time

import cupy as np

from common.util import (plots, progress_bar)

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class Trainer:
    def __init__(self, model, optimizer, train_show_per_iter=32, test_show_per_iter=4):
        self.model, self.optimizer = model, optimizer
        self.train_show_per_iter, self.test_show_per_iter = train_show_per_iter, test_show_per_iter

        self.epochs, self.train_iters, self.test_iters = 0, 0, 0
        self.loss_list, self.train_accuracy_list, self.test_accuracy_list = [], [], []

    def train(self, train_loader, test_loader, epochs=16, batch_size=128, plot=True):
        self.epochs, self.train_iters, self.test_iters = epochs, len(train_loader), len(test_loader)

        for epoch in range(epochs):
            total_loss, train_accu_count, test_accuracys = .0, 0, .0

            train_start_time = time.time()
            for iter, (x_batch, t_batch) in enumerate(train_loader):
                y = self.model.forward(x_batch)

                total_loss += self.model.loss(y, t_batch)
                train_accu_count += np.sum(y.argmax(axis=1) == t_batch).item()

                self.model.backward()
                self.optimizer.update()
                self.optimizer.zero_grad()

                if (iter + 1) % self.train_show_per_iter == 0 or iter + 1 == self.train_iters:
                    interval_iter = self.train_show_per_iter if (iter + 1) % self.train_show_per_iter == 0 \
                        else self.train_iters % self.train_show_per_iter

                    average_loss = total_loss / interval_iter
                    self.loss_list.append(average_loss)
                    train_accuracy = train_accu_count / (batch_size * (iter + 1))

                    self._train_show(epoch, iter, average_loss, train_accuracy, train_start_time)

                    if iter + 1 == self.train_iters:
                        self.train_accuracy_list.append(train_accuracy)
                    total_loss = .0

            test_start_time = time.time()
            for iter, (x_batch, t_batch) in enumerate(test_loader):
                test_accuracys += self.model.val(x_batch, t_batch)

                if (iter + 1) % self.test_show_per_iter == 0 or iter + 1 == self.test_iters:
                    test_accuracy = test_accuracys / (iter + 1)

                    self._test_show(iter, test_accuracy, test_start_time)

                    if iter + 1 == self.test_iters:
                        self.test_accuracy_list.append(test_accuracy)

        if plot:
            plots([self.loss_list], ['train loss'], f'iter * {self.train_show_per_iter}', 'loss')
            plots([self.train_accuracy_list, self.test_accuracy_list], ['train accuracy', 'test accuracy'],
                  f'iter * {self.train_show_per_iter}', 'accuracy')

    def _train_show(self, epoch, iter, average_loss, train_accuracy, start_time):
        message = f'| epoch {epoch + 1:{len(str(self.epochs))}} ' \
                  f'| train ' \
                  f'| iter {iter + 1:{len(str(self.train_iters))}}/{self.train_iters} ' \
                  f'| loss {average_loss:.4f} ' \
                  f'| accuracy {train_accuracy:.4f} ' \
                  f'| time {time.time() - start_time:.2f}s '

        progress_bar(iter, self.train_iters, message=message, break_line=(iter + 1 == self.train_iters))

    def _test_show(self, iter, test_accuracy, start_time):
        epoch_blank = ' ' * len('| epoch ' + str(self.epochs) + ' ')
        iter_blank = ' ' * (len(str(self.train_iters)) - len(str(self.test_iters))) * 2
        loss_blank = ' ' * len(f'| loss 0.0000 ')

        message = f'{epoch_blank}' \
                  f'| test  ' \
                  f'| iter {iter + 1:{len(str(self.test_iters))}}/{self.test_iters} ' \
                  f'{iter_blank}' \
                  f'{loss_blank}' \
                  f'| accuracy {test_accuracy:.4f} ' \
                  f'| time {time.time() - start_time:.2f}s '

        progress_bar(iter, self.test_iters, message=message, break_line=(iter + 1 == self.test_iters))
