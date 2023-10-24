import argparse

import cupy as np

from common.util import to_gpu
from common.models import Linear
from common.optimizer import Adam
from common.trainer import Trainer

from dataset.mnist import load_mnist

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--hidden-size', type=int, nargs='+', default=[100])
    # parser.add_argument('--resume', action='store_true')

    return parser.parse_args()


def print_args(args):
    for key, value in args.items():
        print(f'{key}:{value}', end='  ')
    print()


if __name__ == '__main__':
    opt = vars(parse_opt())
    print_args(opt)
    lr, goal_epoch, batch_size, hidden_size_list = opt['lr'], opt['epochs'], opt['batch_size'], opt['hidden_size']

    (x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
    x_train, t_train = to_gpu(x_train), to_gpu(t_train)

    input_size, class_number = x_train.shape[1], t_train.shape[1]

    model = Linear(input_size, hidden_size_list, class_number)
    optimizer = Adam(model=model, lr=lr)

    trainer = Trainer(model, optimizer)
    trainer.train(x_train, t_train, goal_epoch=goal_epoch, batch_size=batch_size)
    trainer.plot()
