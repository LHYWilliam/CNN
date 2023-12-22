import argparse

import cupy as np

import neolearn

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default=None)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--nosave', action='store_true')
    parser.add_argument('--project', type=str, default='runs/detect/exp')

    return parser.parse_args()


def main(opt):
    weight, data, nosave, project = opt.weight, opt.data, opt.nosave, opt.project

    if not (weight and data):
        return

    data = eval(f'neolearn.datasets.{data}')
    project = neolearn.util.increment_path(project) if not nosave else project

    classes, (_, _), (x_test, t_test) = data.load()
    x_test, t_test = neolearn.util.to_gpu(x_test, t_test)
    test_loader = neolearn.DataLoader(x_test, t_test, 1, shuffle=False)

    checkpoint = neolearn.util.load(weight)
    model = neolearn.model.Model(checkpoint['cfg'])
    model.load(neolearn.util.to_gpu(*checkpoint['params']))

    neolearn.util.print_cfg(model.cfg)

    detector = neolearn.detector.Detector(model, test_loader)
    detector.detect(nosave=nosave, project=project)


if __name__ == '__main__':
    opt = parse_opt()
    neolearn.util.print_args(vars(opt))
    main(opt)
