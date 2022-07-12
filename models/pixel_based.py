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
from utils_tools.utils import uniform_init, orthogonal_init


class ActorModel(nn.Module):
    def __init__(self, state_dim, action_dim, frame_overlay):
        super(ActorModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.frame_overlay = frame_overlay
        self.conv1 = nn.Conv2d(in_channels=self.frame_overlay, out_channels=32,
                               kernel_size=(8, 8), stride=(4, 4))
        self.actv1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(4, 4), stride=(2, 2))
        self.actv2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1))
        self.actv3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.actv4 = nn.ReLU(inplace=True)

        self.fc_state = nn.Sequential(
            nn.Linear(self.state_dim, 200),
            nn.ReLU(inplace=True)
        )

        self.mean_fc1 = nn.Sequential(
            nn.Linear(1024 + 200, 400),
            nn.ReLU(inplace=True))
        self.mean_fc2 = nn.Sequential(
            uniform_init(nn.Linear(400 + 200, 300), a=-3e-3, b=3e-3),
            nn.ReLU(inplace=True))
        self.mean_fc3 = uniform_init(nn.Linear(300, self.action_dim), a=-3e-3, b=3e-3)
        self.mean_fc3act_acc = nn.Sigmoid()
        self.mean_fc3act_ori = nn.Tanh()

        extractor = [self.conv1, self.actv1,
                     self.conv2, self.actv2,
                     self.conv3, self.actv3,
                     self.conv4, self.actv4]
        # extractor = [self.conv1, self.actv1,
        #              self.conv2, self.actv2,
        #              self.conv3, self.actv3]
        self.extractor = nn.Sequential(*extractor)

        layer = [nn.Linear(in_features=9216, out_features=1024),
                 nn.ReLU(inplace=True)]
        self.common_layer = nn.Sequential(*layer)

    def forward(self, state_pixel, state_vect):
        feature_map = self.extractor(state_pixel)
        feature_map = torch.flatten(feature_map, start_dim=1,
                                    end_dim=-1)
        common_vect = self.common_layer(feature_map)

        state_vect = self.fc_state(state_vect)
        common_vect = torch.cat((common_vect, state_vect), dim=-1)
        action_mean = self.mean_fc1(common_vect)
        action_mean = torch.cat((action_mean, state_vect), dim=-1)
        action_mean = self.mean_fc2(action_mean)
        action_mean = self.mean_fc3(action_mean)
        action_mean[..., 0] = self.mean_fc3act_acc(action_mean[..., 0])
        action_mean[..., 1] = self.mean_fc3act_ori(action_mean[..., 1])

        return action_mean


class ActionCriticModel(nn.Module):
    def __init__(self, state_dim, action_dim, frame_overlay):
        super(ActionCriticModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.frame_overlay = frame_overlay
        self.conv1 = nn.Conv2d(in_channels=frame_overlay, out_channels=32,
                               kernel_size=(8, 8), stride=(4, 4))
        self.actv1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(4, 4), stride=(2, 2))
        self.actv2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1))
        self.actv3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.actv4 = nn.ReLU(inplace=True)

        # 特征提取  公用层
        extractor = [self.conv1, self.actv1,
                     self.conv2, self.actv2,
                     self.conv3, self.actv3,
                     self.conv4, self.actv4]
        self.extractor = nn.Sequential(*extractor)

        layer = [nn.Linear(in_features=9216, out_features=1024),
                 nn.ReLU(inplace=True)]
        self.common_layer = nn.Sequential(*layer)
        self.fc_state = nn.Sequential(
            nn.Linear(self.state_dim, 100),
            nn.ReLU(inplace=True)
        )
        self.fc_action = nn.Sequential(
            nn.Linear(self.action_dim, 100),
            nn.ReLU(inplace=True)
        )

        # q1 network
        self.fc_q1 = nn.Sequential(
            orthogonal_init(nn.Linear(1024+100+100, 400), gain=np.sqrt(2)),
            nn.ReLU(inplace=True))
        self.fc2_q1 = nn.Sequential(
            orthogonal_init(nn.Linear(400+100, 64), gain=0.01),
            nn.ReLU(inplace=True),
            orthogonal_init(nn.Linear(64, 1), gain=0.01))

        # q2 network
        self.fc_q2 = nn.Sequential(
            orthogonal_init(nn.Linear(1024+100+100, 400), gain=np.sqrt(2)),
            nn.ReLU(inplace=True))
        self.fc2_q2 = nn.Sequential(
            orthogonal_init(nn.Linear(400+100, 64), gain=0.01),
            nn.ReLU(inplace=True),
            orthogonal_init(nn.Linear(64, 1), gain=0.01))

    def forward(self, state, state_vect, action):
        feature_map = self.extractor(state)
        feature_map = torch.flatten(feature_map, start_dim=1,
                                    end_dim=-1)
        feature_map = torch.flatten(feature_map, start_dim=1, end_dim=-1)
        common_vect = self.common_layer(feature_map)
        state_vect = self.fc_state(state_vect)
        action_vect = self.fc_action(action)
        fusion_vect = torch.cat((common_vect, state_vect, action_vect), dim=-1)
        # q1 network
        q1_critic = self.fc_q1(fusion_vect)
        q1_critic = torch.cat((q1_critic, action_vect), dim=-1)
        q1_critic = self.fc2_q1(q1_critic)
        # q2 network
        q2_critic = self.fc_q2(fusion_vect)
        q2_critic = torch.cat((q2_critic, action_vect), dim=-1)
        q2_critic = self.fc2_q2(q2_critic)
        return q1_critic, q2_critic

    def q_theta1(self, state, state_vect, action):
        feature_map = self.extractor(state)
        feature_map = torch.flatten(feature_map, start_dim=1,
                                    end_dim=-1)
        feature_map = torch.flatten(feature_map, start_dim=1, end_dim=-1)
        common_vect = self.common_layer(feature_map)
        state_vect = self.fc_state(state_vect)
        action_vect = self.fc_action(action)
        fusion_vect = torch.cat((common_vect, state_vect, action_vect), dim=-1)
        # q1 network
        q1_critic = self.fc_q1(fusion_vect)
        q1_critic = torch.cat((q1_critic, action_vect), dim=-1)
        q1_critic = self.fc2_q1(q1_critic)
        return q1_critic


if __name__ == '__main__':
    # model = ActorModel(2*3, 2, frame_overlay=3)
    # model_critic = CriticModel(2*3, 2, frame_overlay=3)
    # x = torch.randn((10, 3, 80, 80))
    # x_vect = torch.rand((10, 2*3))
    # out = model(x, x_vect)
    # from torch.utils.tensorboard import SummaryWriter
    # logger = SummaryWriter(log_dir='./', flush_secs=100)
    # logger.add_graph(model, (x, x_vect))
    # out_critic = model_critic(x)
    critic_model = ActionCriticModel(32, 2, 4)
    pixel = torch.randn((10, 4, 80, 80))
    vect = torch.randn((10, 32))
    act = torch.randn((10, 2)).clamp_(-1, 1)
    target = torch.randn((10, 1))
    opt = torch.optim.Adam(critic_model.parameters(), lr=1e-4)
    loss = torch.nn.MSELoss()
    action_value = critic_model.q_theta1(pixel, vect, act)
    loss_val = loss(action_value, target)
    opt.zero_grad()
    loss_val.backward()
    opt.step()
    critic_model.eval()



