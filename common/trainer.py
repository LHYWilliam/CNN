import time
from collections import deque

import cupy as np

from common.util import (plots, progress_bar, save, to_cpu)

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class Trainer:
    def __init__(self, model, optimizer):
        self.model, self.optimizer = model, optimizer

        self.epochs, self.train_iters, self.test_iters = None, None, None

    def train(self, train_loader, test_loader, epochs=16, batch_size=128,
              train_show=1, test_show=1, nosave=False, noplot=False, early_break=False, project=None):

        self.epochs, self.train_iters, self.test_iters = epochs, len(train_loader), len(test_loader)

        total_iters_loss, train_epochs_accuracy, test_epochs_accuracy = [], [], []
        iters_loss, train_iters_accuracy, test_iters_accuracy = deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)
        test_10epochs_accuracy = deque(maxlen=10)

        for epoch in range(epochs):

            total_loss, train_accu_count = .0, 0
            train_start_time = time.time()

            print('\n     epoch       mod           iter          loss     accuracy        time')

            for iter, (x_batch, t_batch) in enumerate(train_loader):

                y = self.model.forward(x_batch, train=True)

                total_loss += self.model.loss(y, t_batch)
                train_accu_count += np.sum(y.argmax(axis=1) == t_batch).item()

                self.model.backward()
                self.optimizer.update()
                self.optimizer.zero_grad()

                if (iter + 1) % train_show == 0 or iter + 1 == self.train_iters:

                    interval_iter = train_show if (iter + 1) % train_show == 0 \
                        else self.train_iters % train_show

                    total_iters_loss.append(total_loss / interval_iter)

                    iters_loss.append(total_loss / interval_iter)
                    train_iters_accuracy.append(train_accu_count / (batch_size * interval_iter))

                    average_loss = sum(iters_loss) / len(iters_loss)
                    train_average_accuracy = sum(train_iters_accuracy) / len(train_iters_accuracy)

                    self.__train_show(epoch, epochs, iter, average_loss, train_average_accuracy, train_start_time)

                    if iter + 1 == self.train_iters:
                        train_epochs_accuracy.append(train_average_accuracy)

                    total_loss, train_accu_count = .0, .0

            test_total_accuracy, test_average_accuracy, test_best_accuracy = .0, .0, .0
            test_start_time = time.time()
            for iter, (x_batch, t_batch) in enumerate(test_loader):

                test_total_accuracy += self.model.accuracy(x_batch, t_batch)

                if (iter + 1) % test_show == 0 or iter + 1 == self.test_iters:

                    interval_iter = test_show if (iter + 1) % test_show == 0 \
                        else self.train_iters % test_show

                    test_iters_accuracy.append(test_total_accuracy / interval_iter)

                    test_average_accuracy = sum(test_iters_accuracy) / len(train_iters_accuracy)

                    self.__test_show(iter, test_average_accuracy, test_start_time)

                    if iter + 1 == self.test_iters:
                        test_epochs_accuracy.append(test_average_accuracy)
                        test_10epochs_accuracy.append(test_average_accuracy)

                    test_total_accuracy = .0

            if not nosave:
                checkpoint = {
                    'cfg': self.model.cfg,
                    'epoch': epoch + 1,
                    'params': to_cpu(*self.model.params),

                    'lr': self.optimizer.lr,
                    'beta1': self.optimizer.beta1,
                    'beta2': self.optimizer.beta2,
                    'iter': self.optimizer.iter,
                    'm': to_cpu(*self.optimizer.m),
                    'v': to_cpu(*self.optimizer.v)
                }
                save(project / 'last.pkl', checkpoint)
                if test_average_accuracy > test_best_accuracy:
                    save(project / 'best.pkl', checkpoint)

            test_best_accuracy = max(test_average_accuracy, test_best_accuracy)

            if early_break and max(test_10epochs_accuracy) < test_best_accuracy:
                print(f'early break at epoch{epoch + 1}')
                break

        if not noplot:
            plots([total_iters_loss], ['train loss'], f'iter * {train_show}', 'loss')
            plots([train_epochs_accuracy, test_epochs_accuracy], ['train accuracy', 'test accuracy'],
                  f'iter * {train_show}', 'accuracy')

    def __train_show(self, epoch, epochs, iter, average_loss, train_accuracy, start_time):
        epoch_bar = f'{epoch + 1}/{epochs}'
        iter_bar = f'{iter + 1}/{self.train_iters}'

        message = f'{epoch_bar:>10}' \
                  f'     train' \
                  f'{iter_bar:>15}' \
                  f'{average_loss:>14.4f}' \
                  f'{train_accuracy:>13.4f}' \
                  f'{time.time() - start_time:>11.2f}s'

        progress_bar(iter, self.train_iters, message=message, break_line=(iter + 1 == self.train_iters))

    def __test_show(self, iter, test_accuracy, start_time):
        epoch_blank = ' ' * 10
        loss_blank = ' ' * 14

        iter_bar = f'{iter + 1:{len(str(self.test_iters))}}/{self.test_iters}'

        message = f'{epoch_blank}' \
                  f'      test' \
                  f'{iter_bar:>15}' \
                  f'{loss_blank}' \
                  f'{test_accuracy:>13.4f}' \
                  f'{time.time() - start_time:>11.2f}s'

        progress_bar(iter, self.test_iters, message=message, break_line=(iter + 1 == self.test_iters))
