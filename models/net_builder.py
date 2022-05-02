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
        layer_mean = [nn.Linear(self.state_dim, 64),
                      nn.ReLU(inplace=True)]
        self.layer_mean = nn.Sequential(*layer_mean)

        self.mean_fc1 = nn.Linear(64, 64)
        self.mean_fc1act = nn.ReLU(inplace=True)
        self.mean_fc2 = nn.Linear(64, 64)
        self.mean_fc2act = nn.ReLU(inplace=True)
        self.mean_fc3 = nn.Linear(64, self.action_dim)
        nn.init.uniform_(self.mean_fc3.weight, 0, 2e-2)
        self.mean_fc3act = nn.Tanh()

        # self.log_std = nn.Parameter(-1 * torch.ones(action_dim))
        self.log_std = nn.Linear(self.state_dim, 64)
        self.log_std1 = nn.Linear(64, self.action_dim)
        nn.init.uniform_(self.log_std1.weight, -3e-3, 3e-3)

    def forward(self, state):
        mean = self.layer_mean(state)
        action_mean = self.mean_fc1(mean)
        action_mean = self.mean_fc1act(action_mean)
        action_mean = self.mean_fc2(action_mean)
        action_mean = self.mean_fc2act(action_mean)
        action_mean = self.mean_fc3(action_mean)
        action_mean = self.mean_fc3act(action_mean)

        action_std = self.log_std(state)
        action_std = nn.functional.relu(action_std, inplace=True)
        action_std = self.log_std1(action_std)
        action_std = nn.functional.softplus(action_std)
        # 广播机制匹配维度
        """由于是对log_std求exp，所以在计算Normal的时候不需要加1e-8"""
        # action_std = torch.exp(self.log_std)
        dist = Normal(action_mean, action_std)
        action_sample = dist.sample()
        action_sample = torch.clamp(action_sample, -1, 1)
        # try:
        #     action_sample[..., 1] = torch.clamp(action_sample[..., 1], -0.7, 0.7)
        # except IndexError as e:
        #     print('e')
        action_logprob = dist.log_prob(action_sample)

        return action_sample, action_logprob, dist


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
            layer_init(nn.Linear(64, 1)))

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


