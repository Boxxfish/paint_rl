"""
Models for benchmarking.
"""

import torch


class VNet(torch.nn.Module):
    def __init__(self, obs_size: int):
        super(VNet, self).__init__()

        self.dense1 = torch.nn.Linear(obs_size, 8)
        self.out = torch.nn.Linear(8, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.dense1(states)
        x = self.out(x)
        return x