import json

import numpy
import cupy as np

from common.optimizer import Adam
from common.trainer import Trainer
from common.models import Model
from common.dataloader import DataLoader
from common.util import (parse_opt, print_args, print_cfg, load, save, increment_path, to_gpu)

from dataset.mnist import load_mnist

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


def main(opt):
    (cfg, weight, lr, epochs, batch_size, weight_init, nosave, noplot, early_break, project,
     train_show_per_iter, test_show_per_iter, seed) = (opt.cfg, opt.weight, opt.lr, opt.epochs, opt.batch_size,
                                                       opt.weight_init, opt.nosave, opt.noplot, opt.early_break,
                                                       opt.project, opt.train_show_per_iter, opt.test_show_per_iter,
                                                       opt.seed)
    project = increment_path('data/Conv', mkdir=not nosave) if not project and not nosave else project

    numpy.random.seed(seed)
    np.random.seed(seed)

    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    x_train, t_train, x_test, t_test = to_gpu(x_train, t_train, x_test, t_test)
    train_loader = DataLoader(x_train, t_train, batch_size)
    test_loader = DataLoader(x_test, t_test, batch_size)

    if weight:
        weight = load(weight)
        model = Model(weight['cfg'])
        model.load_params(to_gpu(*weight['params']))
        optimizer = Adam(model=model, lr=weight['lr'], beta1=weight['beta1'], beta2=weight['beta2'])
        optimizer.m = to_gpu(*weight['m'])
        optimizer.v = to_gpu(*weight['v'])
        print_cfg(model.cfg)
    elif cfg:
        with open(cfg) as f:
            cfg = json.load(f)
        print_cfg(cfg)
        model = Model(cfg)
        optimizer = Adam(model=model, lr=lr)
    else:
        return

    trainer = Trainer(model, optimizer)
    trainer.train(train_loader, test_loader, epochs=epochs, batch_size=batch_size, train_show=train_show_per_iter,
                  test_show=test_show_per_iter, nosave=nosave, noplot=noplot, early_break=early_break,
                  project=project)


if __name__ == '__main__':
    opt = parse_opt()
    print_args(vars(opt))
    main(opt)
