import yaml
import argparse
from pathlib import Path

import numpy
import neolearn
from neolearn.np import *


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None)
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--noplot', action='store_true')
    parser.add_argument('--project', type=str, default='runs/train/exp')
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


def main(opt):
    (cfg, weight, data, lr, epochs, batch_size, nosave, noplot, project, seed) \
        = (opt.cfg, opt.weight, opt.data, opt.lr, opt.epochs, opt.batch_size,
           opt.nosave, opt.noplot, opt.project, opt.seed)

    if not (cfg or weight) and not data:
        return

    numpy.random.seed(seed)
    np.random.seed(seed)

    data = eval(f'neolearn.datasets.{data}')
    project = neolearn.util.increment_path(project) if not nosave else project

    classes, (x_train, t_train), (x_test, t_test) = data.load()
    train_loader = neolearn.DataLoader(x_train, t_train, batch_size)
    test_loader = neolearn.DataLoader(x_test, t_test, batch_size, shuffle=False)

    if weight:
        checkpoint = neolearn.util.load(weight)
        if neolearn.Config.GPU:
            checkpoint['params'] = neolearn.util.to_gpu(*checkpoint['params'])
            checkpoint['m'] = neolearn.util.to_gpu(*checkpoint['m'])
            checkpoint['v'] = neolearn.util.to_gpu(*checkpoint['v'])
        model = neolearn.model.Model(checkpoint['cfg'])
        model.load(checkpoint['params'])
        optimizer = neolearn.optimizer.Adam(model=model, lr=checkpoint['lr'],
                                            beta1=checkpoint['beta1'], beta2=checkpoint['beta2'])
        optimizer.load([checkpoint['m'], checkpoint['v']])
        neolearn.util.print_cfg(model.cfg)
    elif cfg:
        with open(Path('./neolearn/models') / cfg) as f:
            cfg = yaml.safe_load(f)
        model = neolearn.model.Model(cfg)
        optimizer = neolearn.optimizer.Adam(model=model, lr=lr)
        neolearn.util.print_cfg(cfg)
    else:
        return
    loss = neolearn.loss.SoftmaxWithLoss(model)
    trainer = neolearn.Trainer(model, loss, optimizer, train_loader, test_loader)
    trainer.train(epochs=epochs, batch_size=batch_size, nosave=nosave, noplot=noplot, project=project)


if __name__ == '__main__':
    opt = parse_opt()
    neolearn.util.print_args(vars(opt))
    main(opt)
