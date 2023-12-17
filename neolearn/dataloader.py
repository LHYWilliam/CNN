import numpy
import cupy as np

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class DataLoader:
    def __init__(self, train, test, batch_size, shuffle=True):
        self.train, self.test = train, test
        self.batch_size, self.if_shuffle = batch_size, shuffle

        self.total_size = self.train.shape[0]
        self.length = int(self.total_size / self.batch_size)

        self.iter = -1

        if self.if_shuffle:
            self.shuffle()

    def __len__(self):
        return self.length

    def __getitem__(self, iter):
        begin, end = iter * self.batch_size, (iter + 1) * self.batch_size
        return self.train[begin:end], self.test[begin:end]

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
        self.train, self.test = self.train[index], self.test[index]
