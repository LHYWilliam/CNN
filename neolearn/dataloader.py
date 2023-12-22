import numpy
import cupy as np

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class DataLoader:
    def __init__(self, x, t, batch_size, shuffle=True):
        self.x, self.t = x, t
        self.batch_size, self.if_shuffle = batch_size, shuffle

        self.total_size = self.x.shape[0]
        self.length = int(self.total_size / self.batch_size)

        self.iter = -1

        if self.if_shuffle:
            self.shuffle()

    def __len__(self):
        return self.length

    def __getitem__(self, iter):
        begin, end = iter * self.batch_size, (iter + 1) * self.batch_size
        return self.x[begin:end], self.t[begin:end]

    def __iter__(self):
        return self

    def __next__(self):
        self.iter += 1
        if self.iter < len(self):
            return self[self.iter]
        else:
            self.iter = -1
            if self.if_shuffle:
                self.shuffle()
            raise StopIteration

    def shuffle(self):
        index = numpy.random.permutation(numpy.arange(self.total_size))
        self.x, self.t = self.x[index], self.t[index]
