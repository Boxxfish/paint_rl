"""
Models for copy stroke environment.
"""

import torch

from models.model_utils import get_img_size, init_orthogonal

class VNet(torch.nn.Module):
    def __init__(self, img_size: int):
        super(VNet, self).__init__()

        self.cnn1 = torch.nn.Conv2d(6, 8, 2, 1)
        cnn1_img_size = get_img_size(img_size, self.cnn1)
        self.cnn1_norm = torch.nn.BatchNorm2d(self.cnn1.out_channels)
        self.cnn2 = torch.nn.Conv2d(self.cnn1.out_channels, 16, 2, 1)
        cnn2_img_size = get_img_size(cnn1_img_size, self.cnn2)
        self.cnn2_norm = torch.nn.BatchNorm2d(self.cnn2.out_channels)
        self.flatten = torch.nn.Flatten()
        self.out = torch.nn.Linear(cnn2_img_size**2 * self.cnn2.out_channels, 1)
        self.relu = torch.nn.ReLU()

        init_orthogonal(self)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.cnn1(states)
        x = self.cnn1_norm(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x = self.cnn2_norm(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.out(x)
        return x


class PNet(torch.nn.Module):
    def __init__(self, img_size: int, action_dim: int):
        super(PNet, self).__init__()
        self.action_dim = action_dim
        self.cnn1 = torch.nn.Conv2d(6, 8, 2, 1)
        cnn1_img_size = get_img_size(img_size, self.cnn1)
        self.cnn1_norm = torch.nn.BatchNorm2d(self.cnn1.out_channels)
        self.cnn2 = torch.nn.Conv2d(self.cnn1.out_channels, 16, 2, 1)
        cnn2_img_size = get_img_size(cnn1_img_size, self.cnn2)
        self.cnn2_norm = torch.nn.BatchNorm2d(self.cnn2.out_channels)
        self.flatten = torch.nn.Flatten()
        self.means = torch.nn.Linear(cnn2_img_size**2 * self.cnn2.out_channels, action_dim)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        init_orthogonal(self)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.cnn1(states)
        x = self.cnn1_norm(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x = self.cnn2_norm(x)
        x = self.relu(x)
        x = self.flatten(x)
        means = self.sigmoid(self.means(x))
        return means