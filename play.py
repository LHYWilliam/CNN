import numpy
import cupy as np

import neolearn

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


def main():
    seed = 0
    lr = 0.001
    epochs = 8
    batch_size = 512

    numpy.random.seed(seed)
    np.random.seed(seed)

    (x_train, t_train), (x_test, t_test) = neolearn.dataset.mnist.load(flatten=False)
    x_train, t_train, x_test, t_test = neolearn.util.to_gpu(x_train, t_train, x_test, t_test)
    train_loader = neolearn.DataLoader(x_train, t_train, batch_size)
    test_loader = neolearn.DataLoader(x_test, t_test, batch_size)

    model = neolearn.model.Sequential(
        neolearn.layers.Convolution(1, 16, 3, 1, 1),
        neolearn.layers.BatchNormalization(12544),
        neolearn.layers.ReLu(),
        neolearn.layers.Convolution(16, 16, 3, 1, 1),
        neolearn.layers.BatchNormalization(12544),
        neolearn.layers.ReLu(),
        neolearn.layers.Convolution(16, 16, 2, 2, 0),

        neolearn.layers.Convolution(16, 32, 3, 1, 1),
        neolearn.layers.BatchNormalization(6272),
        neolearn.layers.ReLu(),
        neolearn.layers.Convolution(32, 32, 3, 1, 2),
        neolearn.layers.BatchNormalization(8192),
        neolearn.layers.ReLu(),
        neolearn.layers.Convolution(32, 32, 2, 2, 0),

        neolearn.layers.Convolution(32, 64, 3, 1, 1),
        neolearn.layers.BatchNormalization(4096),
        neolearn.layers.ReLu(),
        neolearn.layers.Convolution(64, 64, 3, 1, 1),
        neolearn.layers.BatchNormalization(4096),
        neolearn.layers.ReLu(),
        neolearn.layers.Convolution(64, 64, 2, 2, 0),

        neolearn.layers.Convolution(64, 1024, 4, 4, 0),

        neolearn.layers.Convolution(1024, 64, 1, 1, 0),
        neolearn.layers.BatchNormalization(64),
        neolearn.layers.ReLu(),

        neolearn.layers.Convolution(64, 10, 1, 1, 0),

        neolearn.layers.Flatten()
    )
    optimizer = neolearn.optimizer.Adam(model=model, lr=lr)
    loss_func = neolearn.loss.SoftmaxWithLoss(model)

    for epoch in range(epochs):
        for iter, (x_batch, t_batch) in enumerate(train_loader):
            y = model.forward(x_batch, train=True)
            loss = loss_func(y, t_batch)
            accuracy = np.sum(np.array(y.argmax(axis=1) == t_batch)).item() / batch_size

            print(f'\repoch {epoch} iter {iter} loss {loss:.4} accuracy {accuracy:.4}', end='')

            loss_func.backward()
            optimizer.update()
            optimizer.zero_grad()


if __name__ == '__main__':
    main()
