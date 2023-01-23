"""
Models for testing PPO.
"""

import torch

from python.model_utils import init_orthogonal


class VNet(torch.nn.Module):
    def __init__(self):
        super(VNet, self).__init__()

        self.dense1 = torch.nn.Linear(1, 8)
        self.dense1_norm = torch.nn.BatchNorm1d(8)
        self.dense2 = torch.nn.Linear(8, 8)
        self.dense2_norm = torch.nn.BatchNorm1d(8)
        self.out = torch.nn.Linear(8, 1)
        self.relu = torch.nn.ReLU()

        init_orthogonal(self)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.dense1(states)
        x = self.dense1_norm(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.dense2_norm(x)
        x = self.relu(x)
        x = self.out(x)
        return x


class PNet(torch.nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        
        self.dense1 = torch.nn.Linear(1, 8)
        self.dense1_norm = torch.nn.BatchNorm1d(8)
        self.dense2 = torch.nn.Linear(8, 8)
        self.dense2_norm = torch.nn.BatchNorm1d(8)
        self.out = torch.nn.Linear(8, 2)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.LogSoftmax(1)

        init_orthogonal(self)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.dense1(states)
        x = self.dense1_norm(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.dense2_norm(x)
        x = self.relu(x)
        x = self.out(x)
        x = self.softmax(x)

        return x