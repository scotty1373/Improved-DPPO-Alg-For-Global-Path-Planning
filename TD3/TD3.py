# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from collections import deque
from copy import deepcopy
from torch.distributions import Normal
from itertools import chain
from models.pixel_based import ActorModel, ActionCriticModel

DISTRIBUTION_INDEX = [0, 0.5]


class TD3:
    def __init__(self, frame_overlay, state_length, action_dim, batch_size, overlay, device, logger=None, root=True):
        self.action_space = (0, 1), (-1, 1)
        self.frame_overlay = frame_overlay
        self.state_length = state_length
        self.state_dim = frame_overlay * state_length
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.overlay = overlay
        self.device = device
        self.logger = logger

        # 初始化actor + 双q网络
        self._init(self.state_dim, self.action_dim, self.frame_overlay)
        self.t = 0
        self.ep = 0
        self.tua = 0.01
        self.lr_actor = 1e-4
        self.lr_critic = 1e-4
        self.start_train = 300
        self.discount_index = 0.95
        self.smooth_regular = 0.2
        self.delay_update = 200
        self.noise = Normal(DISTRIBUTION_INDEX[0], DISTRIBUTION_INDEX[1])

        # optimizer init
        param_chain = chain(self.critic_model1.parameters(), self.critic_model2.parameters())
        self.actor_opt = torch.optim.Adam(params=self.actor_model.parameters(), lr=self.lr_actor)
        self.critic_uni_opt = torch.optim.Adam(params=param_chain, lr=self.lr_critic)

        # loss init
        self.critic_loss = torch.nn.MSELoss()

        # model first hard update
        self.model_hard_update(self.actor_model, self.actor_target)
        self.model_hard_update(self.critic_model, self.critic_target)

    def _init(self, state_dim, action_dim, frame_overlay):
        self.actor_model = ActorModel(state_dim, action_dim, frame_overlay).to(self.device)
        self.actor_target = ActorModel(state_dim, action_dim, frame_overlay).to(self.device)
        self.critic_model = ActionCriticModel(state_dim, action_dim, frame_overlay).to(self.device)
        self.critic_target = ActionCriticModel(state_dim, action_dim, frame_overlay).to(self.device)
        self.memory = deque(maxlen=24000)

    def state_store_memory(self, pixel, vect, action, reward, next_pixel, next_vect, done):
        self.memory.append((pixel, vect, action, reward, next_pixel, next_vect, done, self.t))

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.actor_model(state)
        noise = self.noise.sample()
        logits += noise

        return torch.clamp(logits, self.action_space.min(), self.action_space.max())

    def update(self):
        # 样本抽取
        batch_sample = random.sample(self.memory, self.batch_size)

        # batch数据处理
        pixel, vect, action, reward, next_pixel, next_vect, done, _ = zip(*batch_sample)
        state = np.stack(state, axis=0).squeeze()
        state_t1 = np.stack(state_t1, axis=0).squeeze()
        action = np.concatenate(action).reshape(self.batch_size, 1)
        reward = np.concatenate(reward).reshape(self.batch_size, 1)
        done = np.array(done, dtype='float32').reshape(self.batch_size, 1)

        state = torch.FloatTensor(state)
        state_t1 = torch.FloatTensor(state_t1)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        # critic更新
        self.critic_update(state, action, reward, state_t1, done)

        # actor延迟更新
        if self.t % self.delay_update:
            self.action_update(state)
            with torch.no_grad():
                self.model_soft_update(self.actor_model, self.actor_target)
                self.model_soft_update(self.critic_model1, self.critic_target1)
                self.model_soft_update(self.critic_model2, self.critic_target2)

    # critic theta1为更新主要网络，theta2用于辅助更新
    def action_update(self, state):
        act = self.actor_model(state)
        critic_q1, _ = self.critic_model(state, act)
        actor_loss = - torch.mean(critic_q1)
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        return actor_loss.cpu().detach().item()

    def critic_update(self, state, action, reward, state_t1, done):
        q1_curr, q2_curr = self.critic_model(state, action)

        with torch.no_grad():
            target_action = self.actor_target(state_t1)

            noise = self.noise.sample(sample_shape=target_action.shape)
            epsilon = torch.clamp(noise, min=-self.smooth_regular, max=self.smooth_regular)
            smooth_tg_act = torch.clamp(target_action + epsilon, min=self.action_space.min(),
                                        max=self.action_space.max())

            target_q1 = self.critic_target1(state_t1, smooth_tg_act)
            target_q2 = self.critic_target2(state_t1, smooth_tg_act)
            target_q1q2 = torch.cat([target_q1, target_q2], dim=1)

            # 根据论文附录中遵循deep Q learning，增加终止状态
            td_target = reward + self.discount_index * (1 - done) * torch.min(target_q1q2, dim=1)[0].reshape(self.batch_size, 1)

        loss_q1 = self.critic_loss(td_target, q1_curr)
        loss_q2 = self.critic_loss(td_target, q2_curr)
        loss_critic = loss_q1 + loss_q2

        self.critic_uni_opt.zero_grad()
        loss_critic.backward()
        self.critic_uni_opt.step()

        return loss_critic.cpu().detach().item()

    def save_model(self, file_name):
        checkpoint = {'actor': self.actor_model.state_dict(),
                      'critic1': self.critic_model.state_dict(),
                      'opt_actor': self.actor_opt.state_dict(),
                      'opt_critic': self.critic_uni_opt.state_dict()}
        torch.save(checkpoint, file_name)

    def load_model(self, file_name):
        checkpoint = torch.load(file_name)
        self.actor_model.load_state_dict(checkpoint['actor'])
        self.critic_model.load_state_dict(checkpoint['critic1'])
        self.actor_opt.load_state_dict(checkpoint['opt_actor'])
        self.critic_uni_opt.load_state_dict(checkpoint['opt_critic'])

    @staticmethod
    def model_hard_update(current, target):
        weight_model = deepcopy(current.state_dict())
        target.load_state_dict(weight_model)

    def model_soft_update(self, current, target):
        for target_param, source_param in zip(target.parameters(),
                                              current.parameters()):
            target_param.data.copy_((1 - self.tua) * target_param + self.tua * source_param)
