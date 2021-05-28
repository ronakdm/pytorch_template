import torch.nn as nn
import torch.nn.functional as F


# TODO
class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        pass

    def forward(self, x):
        return F.relu(self.fc(x))
