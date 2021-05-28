import torch.nn as nn
import torch.nn.functional as F


# TODO
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        pass

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        out = self.softmax(x)
        return out
