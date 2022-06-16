# -*-  coding=utf-8 -*-
# @Time : 2022/6/14 10:33
# @Author : Scotty1373
# @File : rnd_model.py
# @Software : PyCharm
import time

import torch
from torch import nn
import torch.functional as F
from utils_tools.utils import layer_init, uniform_init, orthogonal_init
import numpy as np

lr = 1e-3

class RNDModel(nn.Module):
    def __init__(self, state_length):
        super(RNDModel, self).__init__()
        self.state_dim = state_length
        self.update_proportion = 0.25
        self.predict_structure = FeatureExtractor(self.state_dim)
        self.target_structure = FeatureExtractor(self.state_dim)

        # target network不保存梯度
        for layer in self.target_structure.parameters():
            layer.requires_grad = False

    def forward(self, pixel, vect):
        target_vect = self.target_structure(pixel, vect)
        predict_vect = self.predict_structure(pixel, vect)

        return target_vect, predict_vect


class FeatureExtractor(nn.Module):
    def __init__(self, state_dim):
        super(FeatureExtractor, self).__init__()
        assert isinstance(state_dim, int)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8,
                               kernel_size=(8, 8), stride=(4, 4))
        self.actv1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16,
                               kernel_size=(4, 4), stride=(2, 2))
        self.actv2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=(3, 3), stride=(1, 1))
        self.actv3 = nn.LeakyReLU(inplace=True)
        self.fc1 = nn.Linear(1152, 512)
        self.actv4 = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(state_dim, 100)
        self.actv5 = nn.LeakyReLU(inplace=True)
        self.fc3 = nn.Linear(512+100, 400)

        # # 初始化参数
        # for init_iter in self.modules():
        #     if isinstance(init_iter, nn.Conv2d):
        #         nn.init.orthogonal_(init_iter.weight, np.sqrt(2))
        #         init_iter.bias.data.zero_()
        #     elif isinstance(init_iter, nn.Linear):
        #         nn.init.orthogonal_(init_iter.weight, np.sqrt(2))
        #         init_iter.bias.data.zero_()

    def forward(self, pixel, vect):
        pixel = self.conv1(pixel)
        pixel = self.actv1(pixel)
        pixel = self.conv2(pixel)
        pixel = self.actv2(pixel)
        pixel = self.conv3(pixel)
        pixel = self.actv3(pixel)
        pixel = torch.flatten(pixel, start_dim=1, end_dim=-1)
        pixel = self.fc1(pixel)
        pixel = self.actv4(pixel)

        vect = self.fc2(vect)
        vect = self.actv5(vect)
        feature_extractor = self.fc3(torch.cat((pixel, vect), dim=-1))
        return feature_extractor


if __name__ == '__main__':
    rnd = RNDModel(3*6)
    rnd.device = torch.device('cpu')
    x = torch.randn((10, 1, 80, 80))
    y = torch.randn((10, 18))
    pred, targ = rnd(x, y)
    rnd.update(x, y)
    time.time()


