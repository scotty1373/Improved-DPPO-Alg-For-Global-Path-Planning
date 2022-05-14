# -*-  coding=utf-8 -*-
# @Time : 2022/4/27 10:07
# @Author : Scotty1373
# @File : pixel_based.py
# @Software : PyCharm
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class ActorModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16,
                               kernel_size=(8, 8), stride=(4, 4))
        self.actv1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=(4, 4), stride=(2, 2))
        self.actv2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.actv3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.actv4 = nn.ReLU(inplace=True)

        self.mean_fc1 = nn.Linear(512 + self.state_dim, 64)
        self.mean_fc1act = nn.ReLU(inplace=True)
        self.mean_fc2 = nn.Linear(64, self.action_dim)
        nn.init.uniform_(self.mean_fc2.weight, 0, 3e-3)
        self.mean_fc2act = nn.Tanh()

        # self.log_std = nn.Parameter(-1 * torch.ones(action_dim))
        self.log_std = nn.Linear(512 + self.state_dim, 64)
        self.log_std1 = nn.Linear(64, self.action_dim)
        nn.init.normal_(self.log_std1.weight, 0, 3e-4)

        extractor = [self.conv1, self.actv1,
                     self.conv2, self.actv2,
                     self.conv3, self.actv3,
                     self.conv4, self.actv4]
        self.extractor = nn.Sequential(*extractor)

        layer = [nn.Linear(in_features=8192, out_features=512),
                             nn.ReLU(inplace=True)]
        self.common_layer = nn.Sequential(*layer)

    def forward(self, state_pixel, state_vect):
        feature_map = self.extractor(state_pixel)
        feature_map = torch.flatten(feature_map, start_dim=1,
                                    end_dim=-1)
        common_vect = self.common_layer(feature_map)
        common_vect = torch.cat((common_vect, state_vect), dim=-1)
        action_mean = self.mean_fc1(common_vect)
        action_mean = self.mean_fc1act(action_mean)
        action_mean = self.mean_fc2(action_mean)
        action_mean = self.mean_fc2act(action_mean)
        
        action_std = self.log_std(common_vect)
        action_std = nn.functional.relu(action_std, inplace=True)
        action_std = self.log_std1(action_std)
        action_std = nn.functional.softplus(action_std)

        dist = Normal(action_mean, action_std + 1e-8)
        action_sample = dist.sample()
        action_sample = torch.clamp(action_sample, -1, 1)
        action_logprob = dist.log_prob(action_sample)

        return action_sample, action_logprob


class CriticModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16,
                               kernel_size=(8, 8), stride=(4, 4))
        self.actv1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=(4, 4), stride=(2, 2))
        self.actv2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.actv3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.actv4 = nn.ReLU(inplace=True)

        self.fc = nn.Sequential(
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(64, 1)))

        extractor = [self.conv1, self.actv1,
                     self.conv2, self.actv2,
                     self.conv3, self.actv3,
                     self.conv4, self.actv4]
        self.extractor = nn.Sequential(*extractor)

        layer = [nn.Linear(in_features=8192, out_features=256),
                 nn.ReLU(inplace=True)]
        self.common_layer = nn.Sequential(*layer)

    def forward(self, state):
        feature_map = self.extractor(state)
        feature_map = torch.flatten(feature_map, start_dim=1,
                                    end_dim=-1)
        feature_map = torch.flatten(feature_map, start_dim=1, end_dim=-1)
        common_vect = self.common_layer(feature_map)
        common_vect = self.fc(common_vect)

        return common_vect


def layer_init(layer, *, mean=0, std=0.1):
    nn.init.normal_(layer.weight, mean=mean, std=std)
    nn.init.constant_(layer.bias, 0)
    return layer


if __name__ == '__main__':
    model = ActorModel(48, 2)
    model_critic = CriticModel(0, 2)
    x = torch.randn((10, 4, 80, 80))
    x_vect = torch.rand((10, 48))
    out = model(x, x_vect)
    out_critic = model_critic(x)
    from torch.utils.tensorboard import SummaryWriter
    logger = SummaryWriter(log_dir='./', flush_secs=100)
    logger.add_graph(model, (x, x_vect))
    out.shape
