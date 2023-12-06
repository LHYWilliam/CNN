import json
import argparse
from pathlib import Path

import numpy
import cupy as np

import neolearn

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None)
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--weight-init', type=str, default='he')
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--noplot', action='store_true')
    parser.add_argument('--early-break', action='store_true')
    parser.add_argument('--project', type=str, default=None)
    parser.add_argument('--train-show-per-iter', '--train-show', type=int, default=1)
    parser.add_argument('--test-show-per-iter', '--test-show', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


def print_args(args):
    print('\narguments: ', end='')
    for key, value in args.items():
        print(f'{key}:{value}', end='  ', flush=True)


def print_cfg(layers):
    print("\n\nnumber    layer               param")
    for number, layer_param in enumerate(layers):
        layer, param = layer_param.values()
        print(f'{number:<10}{layer:20}{param}')


def main(opt):
    (cfg, weight, lr, epochs, batch_size, nosave, noplot, early_break, project,
     train_show_per_iter, test_show_per_iter, seed) = (opt.cfg, opt.weight, opt.lr, opt.epochs, opt.batch_size,
                                                       opt.nosave, opt.noplot, opt.early_break, opt.project,
                                                       opt.train_show_per_iter, opt.test_show_per_iter, opt.seed)
    project = neolearn.util.increment_path('runs/train', mkdir=not nosave) if (not project and not nosave) else Path(project)

    numpy.random.seed(seed)
    np.random.seed(seed)

    (x_train, t_train), (x_test, t_test) = neolearn.dataset.mnist.load(flatten=False)
    x_train, t_train, x_test, t_test = neolearn.util.to_gpu(x_train, t_train, x_test, t_test)
    train_loader = neolearn.DataLoader(x_train, t_train, batch_size)
    test_loader = neolearn.DataLoader(x_test, t_test, batch_size)

    if weight:
        checkpoint = neolearn.util.load(weight)
        model = neolearn.models.Model(checkpoint['cfg'])
        model.load(neolearn.util.to_gpu(*checkpoint['params']))
        optimizer = neolearn.optimizer.Adam(model=model, lr=checkpoint['lr'], beta1=checkpoint['beta1'], beta2=checkpoint['beta2'])
        optimizer.load([neolearn.util.to_gpu(*checkpoint['m']), neolearn.util.to_gpu(*checkpoint['v'])])
        print_cfg(model.cfg)
    elif cfg:
        with open(Path(Path('./neolearn/models') / cfg)) as f:
            cfg = json.load(f)
        model = neolearn.model.Model(cfg)
        optimizer = neolearn.optimizer.Adam(model=model, lr=lr)
        print_cfg(cfg)
    else:
        return

    trainer = neolearn.Trainer(model, optimizer, train_loader, test_loader)
    trainer.train(epochs=epochs, batch_size=batch_size, train_show=train_show_per_iter,
                  test_show=test_show_per_iter, nosave=nosave, noplot=noplot, early_break=early_break, project=project)


if __name__ == '__main__':
    opt = parse_opt()
    print_args(vars(opt))
    main(opt)
