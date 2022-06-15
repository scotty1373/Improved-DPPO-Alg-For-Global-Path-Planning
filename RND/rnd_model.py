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
    def __init__(self, state_length, device=None):
        super(RNDModel, self).__init__()
        self.state_dim = state_length
        self.device = device
        self.update_proportion = 0.25
        pred_ext_structure = [nn.Conv2d(in_channels=1, out_channels=8,
                                        kernel_size=(8, 8), stride=(4, 4)),
                              nn.LeakyReLU(inplace=True),
                              nn.Conv2d(in_channels=8, out_channels=16,
                                        kernel_size=(4, 4), stride=(2, 2)),
                              nn.LeakyReLU(inplace=True),
                              nn.Conv2d(in_channels=16, out_channels=32,
                                        kernel_size=(3, 3), stride=(1, 1)),
                              nn.LeakyReLU(inplace=True),
                              nn.Flatten(),
                              nn.Linear(1152, 512)]
        pred_sta_structure = [nn.Linear(self.state_dim, 100),
                              nn.LeakyReLU(inplace=True)]
        pred_fusion_structure = [nn.Linear(512 + 100, 400)]

        targ_ext_structure = [nn.Conv2d(in_channels=1, out_channels=8,
                              kernel_size=(8, 8), stride=(4, 4)),
                              nn.LeakyReLU(inplace=True),
                              nn.Conv2d(in_channels=8, out_channels=16,
                                        kernel_size=(4, 4), stride=(2, 2)),
                              nn.LeakyReLU(inplace=True),
                              nn.Conv2d(in_channels=16, out_channels=32,
                                        kernel_size=(3, 3), stride=(1, 1)),
                              nn.LeakyReLU(inplace=True),
                              nn.Flatten(),
                              nn.Linear(1152, 512)]
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

        self.loss_func = nn.MSELoss(reduction='none')

    def forward(self, pixel, vect):
        predict_ext = self.predict_ext(pixel)
        predict_sta = self.predict_sta(vect)
        predict_vect = torch.cat((predict_ext, predict_sta), dim=-1)
        predict_vect = self.predict_fusion(predict_vect)

        target_ext = self.target_ext(pixel)
        target_sta = self.target_sta(vect)
        target_vect = torch.cat((target_ext, target_sta), dim=-1)
        target_vect = self.target_fusion(target_vect)

        return target_vect, predict_vect

    def update(self, pixel, vect):
        self.opt.zero_grad()
        target_vect, predict_vect = self.forward(pixel, vect)
        target_vect = target_vect.detach()
        loss = self.loss_func(predict_vect, target_vect).mean(-1)
        # Proportion of exp used for predictor update
        """from the RND pytorch complete code"""
        """https://github.com/jcwleo/random-network-distillation-pytorch/blob/e383fb95177c50bfdcd81b43e37c443c8cde1d94/agents.py"""
        mask = torch.rand(len(loss)).to(self.device)
        mask = (mask < self.update_proportion).to(self.device)
        loss = (loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
        loss.backward()
        # 未做梯度裁剪
        self.opt.step()


if __name__ == '__main__':
    rnd = RNDModel(3*6)
    rnd.device = torch.device('cpu')
    x = torch.randn((10, 1, 80, 80))
    y = torch.randn((10, 18))
    pred, targ = rnd(x, y)
    rnd.update(x, y)
    time.time()


