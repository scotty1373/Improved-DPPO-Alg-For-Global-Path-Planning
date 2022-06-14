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


class RNDModel(nn.Module):
    def __init__(self, state_dim, frame_overlay):
        super(RNDModel, self).__init__()
        self.state_dim = state_dim
        self.frame_overlay = frame_overlay
        pred_ext_structure = [nn.Conv2d(in_channels=self.frame_overlay, out_channels=32,
                                        kernel_size=(8, 8), stride=(4, 4)),
                              nn.LeakyReLU(inplace=True),
                              nn.Conv2d(in_channels=32, out_channels=64,
                                        kernel_size=(4, 4), stride=(2, 2)),
                              nn.LeakyReLU(inplace=True),
                              nn.Conv2d(in_channels=64, out_channels=64,
                                        kernel_size=(3, 3), stride=(1, 1)),
                              nn.LeakyReLU(inplace=True),
                              nn.Flatten(),
                              nn.Linear(2304, 512)]
        pred_sta_structure = [nn.Linear(self.state_dim, 100),
                              nn.LeakyReLU(inplace=True)]
        pred_fusion_structure = [nn.Linear(512 + 100, 400)]

        targ_ext_structure = [nn.Conv2d(in_channels=self.frame_overlay, out_channels=32,
                              kernel_size=(8, 8), stride=(4, 4)),
                              nn.LeakyReLU(inplace=True),
                              nn.Conv2d(in_channels=32, out_channels=64,
                                        kernel_size=(4, 4), stride=(2, 2)),
                              nn.LeakyReLU(inplace=True),
                              nn.Conv2d(in_channels=64, out_channels=64,
                                        kernel_size=(3, 3), stride=(1, 1)),
                              nn.LeakyReLU(inplace=True),
                              nn.Flatten(),
                              nn.Linear(2304, 512)]
        targ_sta_structure = [nn.Linear(self.state_dim, 100),
                              nn.LeakyReLU(inplace=True)]
        targ_fusion_structure = [nn.Linear(512 + 100, 400)]

        self.predict_ext = nn.Sequential(*pred_ext_structure)
        self.predict_sta = nn.Sequential(*pred_sta_structure)
        self.predict_fusion = nn.Sequential(*pred_fusion_structure)

        self.target_ext = nn.Sequential(*targ_ext_structure)
        self.target_sta = nn.Sequential(*targ_sta_structure)
        self.target_fusion = nn.Sequential(*targ_fusion_structure)

        # 初始化参数
        for init_iter in self.modules():
            if isinstance(init_iter, nn.Conv2d):
                nn.init.orthogonal_(init_iter.weight, np.sqrt(2))
                init_iter.bias.data.zero_()
            elif isinstance(init_iter, nn.Linear):
                nn.init.orthogonal_(init_iter.weight, np.sqrt(2))
                init_iter.bias.data.zero_()

        # target network不保存梯度
        for layer in self.target_ext.parameters():
            layer.requires_grad = False

        for layer in self.target_sta.parameters():
            layer.requires_grad = False

        for layer in self.target_fusion.parameters():
            layer.requires_grad = False

    def forward(self, pixel, state):
        predict_ext = self.predict_ext(pixel)
        predict_sta = self.predict_sta(state)
        predict_vect = torch.cat((predict_ext, predict_sta), dim=-1)
        predict_vect = self.predict_fusion(predict_vect)

        target_ext = self.target_ext(pixel)
        target_sta = self.target_sta(state)
        target_vect = torch.cat((target_ext, target_sta), dim=-1)
        target_vect = self.target_fusion(target_vect)

        return target_vect, predict_vect


if __name__ == '__main__':
    rnd = RNDModel(3*6, 3)
    x = torch.randn((10, 3, 80, 80))
    y = torch.randn((10, 18))
    pred, targ = rnd(x, y)
    time.time()


