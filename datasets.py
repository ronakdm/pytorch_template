import torch
from torch.utils.data import Dataset


# TODO
class MyDataset(Dataset):
    def __init__(self, n, d):
        self.n = n
        self.labels = torch.bernoulli(0.5 * torch.ones(n))
        distributions = [
            torch.distributions.MultivariateNormal(-torch.zeros(d), torch.eye(d)),
            torch.distributions.MultivariateNormal(torch.zeros(d), torch.eye(d)),
        ]
        self.examples = []
        for i in range(n):
            self.examples.append(distributions[self.labels[i].item()].sample())

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.examples[index], self.labels[index]
