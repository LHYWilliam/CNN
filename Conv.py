import json

import numpy
import cupy as np

from common.optimizer import Adam
from common.trainer import Trainer
from common.models import Convolutional
from common.dataloader import DataLoader
from common.util import (parse_opt, print_args, load, save, to_gpu)

from dataset.mnist import load_mnist

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)

if __name__ == '__main__':
    opt = vars(parse_opt())
    print_args(opt)
    cfg, lr, goal_epoch, batch_size, weight_init_std, loads, saves, weight, seed = \
        (opt['cfg'], opt['lr'], opt['epochs'], opt['batch_size'],
         opt['weight_init_std'], opt['loads'], opt['saves'], opt['weight'], opt['seed'])

    numpy.random.seed(seed)
    np.random.seed(seed)

    with open(cfg) as f:
        cfg = json.load(f)

    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    x_train, t_train, x_test, t_test = to_gpu(x_train, t_train, x_test, t_test)
    train_loader = DataLoader(x_train, t_train, batch_size)
    test_loader = DataLoader(x_test, t_test, batch_size)

    channel, input_size, class_number = x_train[1], x_train.shape[2], 10
    if loads:
        model, optimizer = load(weight)
    else:
        model = Convolutional(cfg)
        optimizer = Adam(model=model, lr=lr)

    trainer = Trainer(model, optimizer)
    trainer.train(train_loader, test_loader, epochs=goal_epoch, batch_size=batch_size)

    if saves:
        save(weight, model, optimizer)
