# data.py
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class AddTwoDataSet(Dataset):
    """ 
    Adding dataset as described in https://arxiv.org/abs/1803.01271.
    Implementation based on:
    https://github.com/locuslab/TCN/blob/master/TCN/adding_problem/utils.py
    """
    def __init__(self, N, seq_length):
        self.N = N
        self.seq_length = seq_length

        X_rand = torch.rand(N, 1, seq_length)
        X_add = torch.zeros(N, 1, seq_length)
        self.Y = torch.zeros(N, 1)

        for i in range(N):
            ones = np.random.choice(seq_length, size=2, replace=False)
            X_add[i, 0, ones[0]] = 1
            X_add[i, 0, ones[1]] = 1
            self.Y[i, 0] = X_rand[i, 0, ones[0]] + X_rand[i, 0, ones[1]]

        self.X = torch.cat((X_rand, X_add), dim=1)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()
        return self.X[idx], self.Y[idx]

if __name__ == "__main__":
    # Add two dataset
    print("Add Two dataset: ")
    dataset = AddTwoDataSet(N=1000, seq_length=64)
    loader = DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True)
    dataiter = iter(loader)
    samples, labels = dataiter.next()
    print('Samples : ', samples)
    print('Labels : ', labels)
    print('Shape of samples : ', samples.shape)
    print('Length of dataset: ', dataset.__len__())