"""
DDPG models for copy stroke environment.
"""

import torch

class QNet(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(QNet, self).__init__()

        self.state1 = torch.nn.Linear(state_dim, 16)
        self.state2 = torch.nn.Linear(16, 32)
        self.action1 = torch.nn.Linear(action_dim, 32)
        self.dense1 = torch.nn.Linear(64, 256)
        self.dense2 = torch.nn.Linear(256, 256)
        self.out = torch.nn.Linear(256, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        states = self.state1(states)
        states = self.relu(states)
        states = self.state2(states)
        states = self.relu(states)

        actions = self.action1(actions)
        actions = self.relu(actions)

        x = self.dense1(torch.cat([states, actions], dim=1))
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.out(x)
        return x


class PNet(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(PNet, self).__init__()

        self.dense1 = torch.nn.Linear(state_dim, 256)
        self.dense2 = torch.nn.Linear(256, 256)
        self.out = torch.nn.Linear(256, action_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.dense1(states)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.out(x)
        return x