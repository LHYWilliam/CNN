import cupy as np

from common.models import (Linear)
from common.optimizer import (Adam)
from common.util import (progress_bar)

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class Trainer:
    def __init__(self, model, optimizer):
        self.model, self.optimizer = model, optimizer

    def train(self, x, t, goal_epoch, batch_size, val_per_iter=100):
        total_size = x.shape[0]
        goal_iter = x.shape[0] // batch_size

        for epoch in range(goal_epoch):
            for iter in range(goal_iter):
                batch_mask = np.random.choice(total_size, batch_size)
                x_batch, t_batch = x[batch_mask], t[batch_mask]

                y = self.model.forward(x_batch)
                loss = self.model.loss(x_batch, t_batch)

                self.model.backward()
                params, grads = self.model.params, self.model.grads
                self.optimizer.update(params, grads)

                if val_per_iter and iter % val_per_iter == 0:
                    message = f'| epoch {epoch + 1:{len(str(goal_epoch))}} ' \
                              f'| iter {iter + 1:{len(str(goal_iter))}}/{goal_iter} ' \
                              f'| loss {float(loss):.4f} ' \
                              # f'| time {elapsed_time:.2f}s'
                    progress_bar(iter, goal_iter, message=message)


