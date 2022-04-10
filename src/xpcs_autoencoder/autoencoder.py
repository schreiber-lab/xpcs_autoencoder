# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import torch
from torch import nn, Tensor
from torchvision.models import resnet18

import numpy as np

from .tools import get_size_str

__all__ = ['init_autoencoder', 'AutoEncoder']


def init_autoencoder():
    """
    Initialize the autoencoder model used in the paper.
    :return:
    """
    return AutoEncoder(ResFCEncoder(resnet18(pretrained=False), 32, 512, (3, 3)), SimpleDecoder(32)).cuda()


class ModelMixin:
    def get_params_num(self):
        return sum([np.prod(p.size()) for p in self.parameters()])

    def get_model_size(self):
        return get_size_str(self.get_params_num() * 32)

    def cpu_state_dict(self):
        return OrderedDict([(k, v.to('cpu')) for k, v in self.state_dict().items()])


class ResNetEncoder(nn.Module, ModelMixin):
    def __init__(self, resnet_model, avpool: tuple = (1, 1)):
        super().__init__()
        m = resnet_model

        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.avgpool = nn.AdaptiveAvgPool2d(avpool)

    def forward(self, x):
        x = torch.cat([x, x, x], 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class SimpleDecoder(nn.Module, ModelMixin):
    def __init__(self, latent_dim: int = 256):
        super().__init__()

        self.decoder_input = nn.Linear(latent_dim, 512 * 4)

        hidden_dims = [512, 256, 128, 64, 32]

        modules = [
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i],
                                   hidden_dims[i + 1],
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                nn.LeakyReLU()
            )
            for i in range(len(hidden_dims) - 1)
        ]

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=1,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, z: Tensor) -> Tensor:
        return self.decode(z)


class AutoEncoder(nn.Module, ModelMixin):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ResFCEncoder(ResNetEncoder):
    def __init__(self, resnet, latent_dim: int = 4,
                 hid_dim: int = 512,
                 avpool: tuple = (2, 2)):
        super().__init__(resnet, avpool)

        self.fc1 = nn.Linear(int(256 * np.prod(avpool)), hid_dim)
        self.leaky_relu = nn.LeakyReLU(0.01, inplace=True)
        self.fc2 = nn.Linear(hid_dim, latent_dim)

    def forward(self, x):
        x = super().forward(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        return x
