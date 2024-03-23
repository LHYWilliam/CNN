import argparse

import cupy
import numpy

import neolearn


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--project', type=str, default='runs/detect/exp')

    return parser.parse_args()


def main(opt):
    weight, data, batch_size, nosave, project = opt.weight, opt.data, opt.batch_size, opt.nosave, opt.project

    if not (weight and data):
        return

    numpy.random.seed(0)
    if neolearn.Config.GPU:
        cupy.random.seed(0)

    data = eval(f'neolearn.datasets.{data}')
    project = neolearn.util.increment_path(project) if not nosave else project

    _, (_, _), (x_test, t_test) = data.load()
    test_loader = neolearn.DataLoader(x_test, t_test, batch_size, shuffle=False)

    checkpoint = neolearn.util.load(weight)
    model = neolearn.model.Model(checkpoint['cfg'])
    if neolearn.Config.GPU:
        checkpoint['params'] = neolearn.util.to_gpu(*checkpoint['params'])
    model.load(checkpoint['params'])

    neolearn.util.print_cfg(model.cfg)

    detector = neolearn.Detector(model, test_loader)
    detector.detect(batch_size=batch_size, nosave=nosave, project=project)


if __name__ == '__main__':
    opt = parse_opt()
    neolearn.util.print_args(vars(opt))
    main(opt)
