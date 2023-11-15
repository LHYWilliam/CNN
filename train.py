import json

from pathlib import Path

import numpy
import cupy as np

from common.optimizer import Adam
from common.trainer import Trainer
from common.models import Model
from common.dataloader import DataLoader
from common.util import (parse_opt, print_args, print_cfg, load, increment_path, to_gpu)

from dataset.mnist import load_mnist

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


def main(opt):
    (cfg, weight, lr, epochs, batch_size, nosave, noplot, early_break, project,
     train_show_per_iter, test_show_per_iter, seed) = (opt.cfg, opt.weight, opt.lr, opt.epochs, opt.batch_size,
                                                       opt.nosave, opt.noplot, opt.early_break, opt.project,
                                                       opt.train_show_per_iter, opt.test_show_per_iter, opt.seed)
    project = increment_path('runs/train', mkdir=not nosave) if (not project and not nosave) else Path(project)

    numpy.random.seed(seed)
    np.random.seed(seed)

    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    x_train, t_train, x_test, t_test = to_gpu(x_train, t_train, x_test, t_test)
    train_loader = DataLoader(x_train, t_train, batch_size)
    test_loader = DataLoader(x_test, t_test, batch_size)

    if weight:
        checkpoint = load(weight)
        model = Model(checkpoint['cfg'])
        model.load(to_gpu(*checkpoint['params']))
        optimizer = Adam(model=model, lr=checkpoint['lr'], beta1=checkpoint['beta1'], beta2=checkpoint['beta2'])
        optimizer.load([to_gpu(*checkpoint['m']), to_gpu(*checkpoint['v'])])
        print_cfg(model.cfg)
    elif cfg:
        with open(cfg) as f:
            cfg = json.load(f)
        model = Model(cfg)
        optimizer = Adam(model=model, lr=lr)
        print_cfg(cfg)
    else:
        return

    trainer = Trainer(model, optimizer, train_loader, test_loader)
    trainer.train(epochs=epochs, batch_size=batch_size, train_show=train_show_per_iter,
                  test_show=test_show_per_iter, nosave=nosave, noplot=noplot, early_break=early_break, project=project)


if __name__ == '__main__':
    opt = parse_opt()
    print_args(vars(opt))
    main(opt)
