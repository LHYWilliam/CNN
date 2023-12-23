import time

import cupy as np

from neolearn.util import (progress_bar)

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class Detector:
    def __init__(self, model, test_loader):
        self.model, self.test_loader = model, test_loader
        self.test_iters = len(test_loader)

    def detect(self, batch_size=1, nosave=False, project=None):
        print('\n       mod           iter     accuracy        time')

        accuracy, start_time = 0., time.time()
        for iter, (x_batch, t_batch) in enumerate(self.test_loader):
            y = self.model.forward(x_batch, train=False)

            # print(f'{y.argmax(axis=1)} -> {t_batch}')

            accuracy += np.sum(np.array(y.argmax(axis=1) == t_batch)).item() / batch_size

            self._test_show(iter, accuracy / (iter + 1), start_time)

        # TODO: save

    def _test_show(self, iter, accuracy, start_time):
        iter_bar = f'{iter + 1:{len(str(self.test_iters))}}/{self.test_iters}'

        message = f'      test' \
                  f'{iter_bar:>15}' \
                  f'{accuracy:>13.4f}' \
                  f'{time.time() - start_time:>11.2f}s'

        progress_bar(iter, self.test_iters, message=message, break_line=(iter + 1 == self.test_iters))
