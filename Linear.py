import argparse

import numpy
import cupy as np

from common.util import (parse_opt,print_args, load, save, to_gpu)
from common.models import Linear
from common.optimizer import Adam
from common.trainer import Trainer
from common.dataloader import DataLoader

from dataset.mnist import load_mnist

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


if __name__ == '__main__':
    opt = vars(parse_opt())
    print_args(opt)
    lr, goal_epoch, batch_size, hidden_size_list, weight_init_std, loads, saves, weight, seed = \
        (opt['lr'], opt['epochs'], opt['batch_size'], opt['hidden_size'],
         opt['weight_init_std'], opt['loads'], opt['saves'], opt['weight'], opt['seed'])

    numpy.random.seed(seed)
    np.random.seed(seed)

    (x_train, t_train), (x_test, t_test) = load_mnist()
    x_train, t_train, x_test, t_test = to_gpu(x_train, t_train, x_test, t_test)
    train_loader = DataLoader(x_train, t_train, batch_size)
    test_loader = DataLoader(x_test, t_test, batch_size)

    input_size, class_number = x_train.shape[1], 10
    if loads:
        model, optimizer = load(weight)
    else:
        model = Linear(input_size, hidden_size_list, class_number, weight_init=weight_init_std)
        optimizer = Adam(model=model, lr=lr)

    trainer = Trainer(model, optimizer)
    trainer.train(train_loader, test_loader, epochs=goal_epoch, batch_size=batch_size)

    if saves:
        save(weight, model, optimizer)
