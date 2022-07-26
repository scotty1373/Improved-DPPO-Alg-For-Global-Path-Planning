# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from collections import deque
from copy import deepcopy
from torch.distributions import Normal
from itertools import chain
from models.pixel_based import ActorModel, ActionCriticModel
from utils_tools.utils import RunningMeanStd, cut_requires_grad

DISTRIBUTION_INDEX = [0, 0.3]
ENV_RESET_BOUND = [50, 150, 300, 500]


class TD3:
    def __init__(self, frame_overlay, state_length, action_dim, batch_size, overlay, device, train=True, logger=None, *, main_process=True):
        self.action_space = np.array((-1, 1))
        self.frame_overlay = frame_overlay
        self.state_length = state_length
        self.state_dim = frame_overlay * state_length
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.overlay = overlay
        self.device = device
        self.train = train
        self.logger = logger
        self.main_process = main_process

        # 初始化actor + 双q网络
        self._init(self.state_dim, self.action_dim, self.frame_overlay, self.main_process)
        self.t = 0
        self.ep = 0
        self.tua = 0.01
        self.lr_actor = 1e-5
        self.lr_critic = 1e-5
        self.start_train = 2000
        self.discount_index = 0.95
        self.smooth_regular = 0.2
        self.delay_update = 5
        self.noise = Normal(DISTRIBUTION_INDEX[0], DISTRIBUTION_INDEX[1])
        self.target_model_regular_noise = Normal(0, 0.2)

        # optimizer init
        self.actor_opt = torch.optim.Adam(params=self.actor_model.parameters(), lr=self.lr_actor)
        self.critic_uni_opt = torch.optim.Adam(params=self.critic_model.parameters(), lr=self.lr_critic)

        # reward rms
        self.rwd_rms = RunningMeanStd()

        # loss init
        self.critic_loss = torch.nn.MSELoss()

        # loss history
        self.actor_loss_history = 0.0
        self.critic_loss_history = 0.0

        # 环境复杂度提升标志
        self.complex = 0

    def _init(self, state_dim, action_dim, frame_overlay, root):
        self.actor_model = ActorModel(state_dim, action_dim, frame_overlay).to(self.device)
        self.critic_model = ActionCriticModel(state_dim, action_dim, frame_overlay).to(self.device)
        if root:
            self.actor_target = ActorModel(state_dim, action_dim, frame_overlay).to(self.device)
            self.critic_target = ActionCriticModel(state_dim, action_dim, frame_overlay).to(self.device)
            # model first hard update
            self.model_hard_update(self.actor_model, self.actor_target)
            self.model_hard_update(self.critic_model, self.critic_target)
            # target model requires_grad设为False
            cut_requires_grad(self.actor_target.parameters())
            cut_requires_grad(self.critic_target.parameters())

    def reset_noise(self):
        if self.ep <= 50:
            self.noise = Normal(DISTRIBUTION_INDEX[0], DISTRIBUTION_INDEX[1] * 0.98 ** (self.ep-50) + 0.05)
            self.target_model_regular_noise = Normal(0, 0.2)
        elif 50 < self.ep <= 150:
            self.noise = Normal(DISTRIBUTION_INDEX[0], DISTRIBUTION_INDEX[1] * 0.98 ** (self.ep-150) + 0.05)
            self.target_model_regular_noise = Normal(0, 0.2)
        elif 150 < self.ep <= 300:
            self.noise = Normal(DISTRIBUTION_INDEX[0], DISTRIBUTION_INDEX[1] * 0.98 ** (self.ep-300) + 0.05)
            self.target_model_regular_noise = Normal(0, 0.1)
        elif 300 < self.ep <= 500:
            self.noise = Normal(DISTRIBUTION_INDEX[0], DISTRIBUTION_INDEX[1] * 0.98 ** (self.ep-500) + 0.05)
            self.target_model_regular_noise = Normal(0, 0.05)

    def get_action(self, pixel_state, vect_state):
        pixel = torch.FloatTensor(pixel_state).to(self.device)
        vect = torch.FloatTensor(vect_state).to(self.device)
        logits = self.actor_model(pixel, vect)
        if self.train:
            # acc 动作裁剪
            logits[..., 0] = (logits[..., 0] + self.noise.sample()).clamp_(min=0, max=1)
            # ori 动做裁剪
            logits[..., 1] = (logits[..., 1] + self.noise.sample()).clamp_(min=self.action_space.min(),
                                                                           max=self.action_space.max())
        else:
            logits[..., 0] = logits[..., 0].clamp_(min=0, max=1)
            logits[..., 1] = logits[..., 1].clamp_(min=self.action_space.min(), max=self.action_space.max())
        return logits.detach().cpu().numpy()

    def update(self, replay_buffer):
        if replay_buffer.size < self.start_train:
            return
        # batch数据处理
        pixel, next_pixel, vect, next_vect, reward, action, done = replay_buffer.get_batch(self.batch_size)

        # critic更新
        self.critic_update(pixel, vect, action, next_pixel, next_vect, reward, done)

        # actor延迟更新
        if self.t % self.delay_update == 0:
            self.action_update(pixel_state=pixel, vect_state=vect)
            with torch.no_grad():
                self.model_soft_update(self.actor_model, self.actor_target)
                self.model_soft_update(self.critic_model, self.critic_target)

        self.logger.add_scalar(tag='actor_loss',
                               scalar_value=self.actor_loss_history,
                               global_step=self.t)
        self.logger.add_scalar(tag='critic_loss',
                               scalar_value=self.critic_loss_history,
                               global_step=self.t)

    # critic theta1为更新主要网络，theta2用于辅助更新
    def action_update(self, pixel_state, vect_state):
        act = self.actor_model(pixel_state, vect_state)
        critic_q1 = self.critic_model.q_theta1(pixel_state, vect_state, act)
        actor_loss = - torch.mean(critic_q1)
        self.critic_uni_opt.zero_grad()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        self.actor_loss_history = actor_loss.item()

    def critic_update(self, pixel_state, vect_state, action, next_pixel_state, next_vect_state, reward, done):
        with torch.no_grad():
            target_action = self.actor_target(next_pixel_state, next_vect_state)

            noise = self.target_model_regular_noise.sample(sample_shape=target_action.shape).to(self.device)
            epsilon = torch.clamp(noise, min=-self.smooth_regular, max=self.smooth_regular)
            smooth_tg_act = target_action + epsilon
            smooth_tg_act[..., 0].clamp_(min=0, max=self.action_space.max())
            smooth_tg_act[..., 1].clamp_(min=self.action_space.min(), max=self.action_space.max())

            target_q1, target_q2 = self.critic_target(next_pixel_state, next_vect_state, smooth_tg_act)
            target_q1q2 = torch.cat([target_q1, target_q2], dim=1)

            # 根据论文附录中遵循deep Q learning，增加终止状态
            td_target = reward + self.discount_index * (1 - done) * torch.min(target_q1q2, dim=1)[0].reshape(-1, 1)

        q1_curr, q2_curr = self.critic_model(pixel_state, vect_state, action)
        loss_q1 = self.critic_loss(q1_curr, td_target)
        loss_q2 = self.critic_loss(q2_curr, td_target)
        loss_critic = loss_q1 + loss_q2

        self.critic_uni_opt.zero_grad()
        loss_critic.backward()
        self.critic_uni_opt.step()
        self.critic_loss_history = loss_critic.item()

    def save_model(self, file_name):
        checkpoint = {'actor': self.actor_model.state_dict(),
                      'critic': self.critic_model.state_dict(),
                      'opt_actor': self.actor_opt.state_dict(),
                      'opt_critic': self.critic_uni_opt.state_dict()}
        torch.save(checkpoint, file_name)

    def load_model(self, file_name):
        checkpoint = torch.load(file_name)
        self.actor_model.load_state_dict(checkpoint['actor'])
        self.critic_model.load_state_dict(checkpoint['critic'])
        self.actor_opt.load_state_dict(checkpoint['opt_actor'])
        self.critic_uni_opt.load_state_dict(checkpoint['opt_critic'])

    def model_hard_update(self, current, target):
        weight_model = deepcopy(current.state_dict())
        target.load_state_dict(weight_model)

    def model_soft_update(self, current, target):
        for target_param, source_param in zip(target.parameters(),
                                              current.parameters()):
            target_param.data.copy_((1 - self.tua) * target_param + self.tua * source_param)
