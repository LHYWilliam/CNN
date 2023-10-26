import cupy as np

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)


class DataLoader:
    def __init__(self, train, test, batch_size):
        self.train, self.test = train, test
        self.batch_size = batch_size

    def __len__(self):
        return self.train.shape[0]

    def __getitem__(self, position):
        return (self.train[position * self.batch_size:(position + 1) * self.batch_size],
                self.test[position * self.batch_size:(position + 1) * self.batch_size])
