import torch
from torch.utils.data import Dataset


# TODO
class MyDataset(Dataset):
    def __init__(self, n, d):
        self.n = n
        self.examples = torch.rand(n, d)
        self.labels = torch.rand(n)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.examples[index], self.labels[index]
