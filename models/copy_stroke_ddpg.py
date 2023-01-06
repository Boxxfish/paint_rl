"""
DDPG models for copy stroke environment.
"""

import torch

from models.model_utils import get_img_size

class QNet(torch.nn.Module):
    def __init__(self, img_size: int, action_dim: int):
        super(QNet, self).__init__()

        self.state1 = torch.nn.Conv2d(6, 8, 3, 2)
        state1_img_size = get_img_size(img_size, self.state1)
        self.state1_norm = torch.nn.BatchNorm2d(self.state1.out_channels)
        self.state2 = torch.nn.Conv2d(self.state1.out_channels, 16, 3, 2)
        state2_img_size = get_img_size(state1_img_size, self.state2)
        self.state2_norm = torch.nn.BatchNorm2d(self.state2.out_channels)
        self.state_flatten = torch.nn.Flatten()
        
        self.action1 = torch.nn.Linear(action_dim, 32)
        self.action1_norm = torch.nn.BatchNorm1d(self.action1.out_features)
        self.action2 = torch.nn.Linear(self.action1.out_features, 64)
        self.action2_norm = torch.nn.BatchNorm1d(self.action2.out_features)
        
        self.dense1 = torch.nn.Linear(self.action2.out_features + state2_img_size**2 * self.state2.out_channels, 256)
        self.dense1_norm = torch.nn.BatchNorm1d(self.dense1.out_features)
        self.dense2 = torch.nn.Linear(self.dense1.out_features, 256)
        self.dense2_norm = torch.nn.BatchNorm1d(self.dense2.out_features)
        self.out = torch.nn.Linear(self.dense2.out_features, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        states = self.state1(states)
        states = self.relu(states)
        states = self.state2(states)
        states = self.relu(states)
        states = self.state_flatten(states)
        states = self.relu(states)

        actions = self.action1(actions)
        actions = self.relu(actions)
        actions = self.action2(actions)
        actions = self.relu(actions)

        x = self.dense1(torch.cat([states, actions], dim=1))
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.out(x)
        return x


class PNet(torch.nn.Module):
    def __init__(self, img_size: int, action_dim: int):
        super(PNet, self).__init__()
        self.cnn1 = torch.nn.Conv2d(6, 8, 3, 2)
        cnn1_img_size = get_img_size(img_size, self.cnn1)
        self.cnn1_norm = torch.nn.BatchNorm2d(self.cnn1.out_channels)
        self.cnn2 = torch.nn.Conv2d(self.cnn1.out_channels, 16, 3, 2)
        cnn2_img_size = get_img_size(cnn1_img_size, self.cnn2)
        self.cnn2_norm = torch.nn.BatchNorm2d(self.cnn2.out_channels)
        self.flatten = torch.nn.Flatten()
        self.out = torch.nn.Linear(cnn2_img_size**2 * self.cnn2.out_channels, action_dim)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.cnn1(states)
        x = self.cnn1_norm(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x = self.cnn2_norm(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x