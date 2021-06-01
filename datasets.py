import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, n, d, mean_scale=1.0, cov_scales=[1.0, 0.5]):
        self.n = n
        self.labels = torch.bernoulli(0.5 * torch.ones(n)).long()
        distributions = [
            torch.distributions.MultivariateNormal(
                -mean_scale * torch.ones(d),
                covariance_matrix=cov_scales[0] * torch.eye(d),
            ),
            torch.distributions.MultivariateNormal(
                mean_scale * torch.ones(d),
                covariance_matrix=cov_scales[1] * torch.eye(d),
            ),
        ]
        self.examples = []
        for i in range(n):
            self.examples.append(distributions[int(self.labels[i])].sample())

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.examples[index], self.labels[index]
