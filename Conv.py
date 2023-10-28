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
    opt = parse_opt()
    print_args(vars(opt))
    cfg, weight, lr, epochs, batch_size, weight_init, nosave, train_show_per_iter, test_show_per_iter, seed = \
        (opt.cfg, opt.weight, opt.lr, opt.epochs, opt.batch_size,
         opt.weight_init, opt.nosave, opt.train_show_per_iter, opt.test_show_per_iter, opt.seed)

    numpy.random.seed(seed)
    np.random.seed(seed)

    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    x_train, t_train, x_test, t_test = to_gpu(x_train, t_train, x_test, t_test)
    train_loader = DataLoader(x_train, t_train, batch_size)
    test_loader = DataLoader(x_test, t_test, batch_size)

    channel, input_size, class_number = x_train[1], x_train.shape[2], 10
    if weight:
        model, optimizer = load(weight)
    elif cfg:
        with open(cfg) as f:
            cfg = json.load(f)
        model = Convolutional(cfg)
        optimizer = Adam(model=model, lr=lr)
    else:
        exit(code=1)

    trainer = Trainer(model, optimizer)
    trainer.train(train_loader, test_loader, epochs=epochs, batch_size=batch_size,
                  train_show_per_iter=train_show_per_iter, test_show_per_iter=test_show_per_iter)

    if not nosave:
        save(weight, model, optimizer)
