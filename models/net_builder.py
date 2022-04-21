# -*-  coding=utf-8 -*-
# @Time : 2022/4/20 9:46
# @Author : Scotty1373
# @File : net_builder.py
# @Software : PyCharm
import time

import torch
from torch import nn
from torch.distributions import Normal
import numpy as np

class ActorModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        layer_common = [nn.Linear(self.state_dim, 256),
                        nn.ReLU(inplace=True)]
        self.layer_common = nn.Sequential(*layer_common)

        self.mean_fc1 = nn.Linear(256, 128)
        self.mean_fc1act = nn.ReLU(inplace=True)
        self.mean_fc2 = nn.Linear(128, 64)
        self.mean_fc2act = nn.ReLU(inplace=True)
        self.mean_fc3 = nn.Linear(64, self.action_dim)
        nn.init.uniform_(self.mean_fc3.weight, -3e-3, 3e-3)
        self.mean_fc3act = nn.Tanh()

        self.std_fc1 = nn.Linear(256, 128)
        self.std_fc1act = nn.ReLU(inplace=True)
        self.std_fc2 = nn.Linear(128, self.action_dim)
        self.std_fc2act = nn.Softplus()

    def forward(self, state):
        common = self.layer_common(state)
        action_mean = self.mean_fc1(common)
        action_mean = self.mean_fc1act(action_mean)
        action_mean = self.mean_fc2(action_mean)
        action_mean = self.mean_fc2act(action_mean)
        action_mean = self.mean_fc3(action_mean)
        action_mean = self.mean_fc3act(action_mean)

        action_std = self.std_fc1(common)
        action_std = self.std_fc1act(action_std)
        action_std = self.std_fc2(action_std)
        action_std = self.std_fc2act(action_std)

        return action_mean, action_std

    def get_action(self, state):
        action_mean, action_cov = self.forward(state)
        dist = Normal(action_mean, action_cov)
        action_sample = dist.sample()
        return action_sample


class CriticModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, 256)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(256, 64)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(64, 1)),
            nn.Softplus())

    def forward(self, state):
        value = self.fc(state)
        return value

def layer_init(layer, *, mean=0, std=0.1):
    nn.init.normal_(layer.weight, mean=mean, std=std)
    nn.init.constant_(layer.bias, 0)
    return layer


if __name__ == '__main__':
    actor = ActorModel(8, 2)
    critic = CriticModel(8, 2)
    obs_state = torch.randn(10, 8)
    mean_vec, cov_mat = actor(obs_state)
    val = critic(obs_state)
    action = actor.get_action(obs_state)


