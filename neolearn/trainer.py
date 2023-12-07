from calendar import c
import time

import cupy as np

from neolearn.caculate import Caculate
from neolearn.util import (plots, progress_bar, save, to_cpu)

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class Trainer:
    def __init__(self, model, loss, optimizer, train_loader, test_loader):
        self.train_iters, self.test_iters = None, None
        self.model, self.loss, self.optimizer = model, loss, optimizer
        self.train_loader, self.test_loader = train_loader, test_loader

    def train(self, epochs=16, batch_size=128, nosave=False, noplot=False, project=None):
        self.train_iters, self.test_iters = len(self.train_loader), len(self.test_loader)
        caculate = Caculate(self.train_iters, self.test_iters)

        for epoch in range(epochs):
            print('\n     epoch       mod           iter          loss     accuracy        time')

            start_time = time.time()
            for iter, (x_batch, t_batch) in enumerate(self.train_loader):

                y = self.model.forward(x_batch, train=True)

                loss = self.loss(y, t_batch)
                accuracy = np.sum(y.argmax(axis=1) == t_batch).item() / batch_size
                caculate.train_add(loss, accuracy)

                self.loss.backward()
                self.optimizer.update()
                self.optimizer.zero_grad()

                self.__train_show(epoch, epochs, iter, caculate.average_loss, caculate.train_average_accuracy, start_time)

            start_time = time.time()
            for iter, (x_batch, t_batch) in enumerate(self.test_loader):

                accuracy = self.model.accuracy(x_batch, t_batch)
                caculate.test_add(accuracy)

                self.__test_show(iter, caculate.test_average_accuracy, start_time)

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
                if caculate.test_average_accuracy > caculate.test_best_accuracy:
                    save(project / 'best.pkl', checkpoint)

        if not noplot:
            plots([caculate.loss], ['train loss'], 'iter', 'loss')
            plots([caculate.train_epochs_accuracy, caculate.test_epochs_accuracy], ['train accuracy', 'test accuracy'],
                  'iter', 'accuracy')

    def __train_show(self, epoch, epochs, iter, loss, accuracy, start_time):
        epoch_bar = f'{epoch + 1}/{epochs}'
        iter_bar = f'{iter + 1}/{self.train_iters}'

        message = f'{epoch_bar:>10}' \
                  f'     train' \
                  f'{iter_bar:>15}' \
                  f'{loss:>14.4f}' \
                  f'{accuracy:>13.4f}' \
                  f'{time.time() - start_time:>11.2f}s'

        progress_bar(iter, self.train_iters, message=message, break_line=(iter + 1 == self.train_iters))

    def __test_show(self, iter, accuracy, start_time):
        epoch_blank = ' ' * 10
        loss_blank = ' ' * 14

        iter_bar = f'{iter + 1:{len(str(self.test_iters))}}/{self.test_iters}'

        message = f'{epoch_blank}' \
                  f'      test' \
                  f'{iter_bar:>15}' \
                  f'{loss_blank}' \
                  f'{accuracy:>13.4f}' \
                  f'{time.time() - start_time:>11.2f}s'

        progress_bar(iter, self.test_iters, message=message, break_line=(iter + 1 == self.test_iters))
