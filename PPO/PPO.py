import numpy
import torch
from models.net_builder import ActorModel, CriticModel
from torch.distributions import Normal
from collections import deque
from skimage.color import rgb2gray
from PIL import Image
import numpy as np
import copy

LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-4
DECAY = 0.99
EPILSON = 0.2
torch.autograd.set_detect_anomaly(True)


class PPO:
    def __init__(self, state_dim, action_dim, batch_size):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.batch_size = batch_size

        # model build
        self._init(self.state_dim, self.action_dim, self.batch_size)

        # optimizer initialize
        self.lr_actor = LEARNING_RATE_ACTOR
        self.lr_critic = LEARNING_RATE_CRITIC
        self.decay_index = DECAY
        self.epilson = EPILSON
        self.c_loss = torch.nn.MSELoss()
        self.c_opt = torch.optim.Adam(params=self.v.parameters(), lr=self.lr_critic)
        self.a_opt = torch.optim.Adam(params=self.pi.parameters(), lr=self.lr_actor)

        # training configuration
        self.update_actor_epoch = 3
        self.update_critic_epoch = 3
        self.history_critic = 0
        self.history_actor = 0
        self.t = 0
        self.ep = 0

    def _init(self, state_dim, action_dim, train_batch):
        self.pi = ActorModel(state_dim, action_dim)
        self.piold = ActorModel(state_dim, action_dim)
        self.v = CriticModel(state_dim, action_dim)
        self.memory = deque(maxlen=train_batch)

    def get_action(self, obs_):
        obs_ = torch.Tensor(copy.deepcopy(obs_))
        self.pi.eval()
        with torch.no_grad():
            ac_mean, ac_sigma = self.pi(obs_)

            # 增加1e-8防止正态分布计算时除法越界
            dist = Normal(ac_mean.cpu().detach(), ac_sigma.cpu().detach() + 1e-8)

            action_sample = dist.sample()
            prob_acc = torch.clamp(action_sample[..., 0], -1, 1).cpu().detach().numpy()
            prob_ori = torch.clamp(action_sample[..., 1], -1, 1).cpu().detach().numpy()
            # log_prob = dist.log_prob(torch.stack((prob_acc, prob_ori), dim=0))
        self.pi.train()

        return prob_acc, prob_ori, ac_mean.cpu().detach().numpy()

    def state_store_memory(self, s, acc, ori, r, logprob_acc, logprob_ori):
        self.memory.append((s, acc, ori, r, logprob_acc, logprob_ori))

    # 计算reward衰减，根据马尔可夫过程，从最后一个reward向前推
    def decayed_reward(self, singal_state_frame, reward_):
        decayed_rd = []
        with torch.no_grad():
            state_frame = torch.Tensor(singal_state_frame)
            value_target = self.v(state_frame).detach().numpy()
            for rd_ in reward_[::-1]:
                value_target = rd_ + value_target * self.decay_index
                decayed_rd.append(value_target)
            decayed_rd.reverse()
        return decayed_rd

    # 计算actor更新用的advantage value
    def advantage_calcu(self, decay_reward, state_t):
        with torch.no_grad():
            state_t = torch.Tensor(state_t)
            critic_value_ = self.v(state_t)
            d_reward = torch.Tensor(decay_reward)
            advantage = d_reward - critic_value_
        return advantage

    # 计算critic更新用的 Q(s, a)和 V(s)
    def critic_update(self, state_t1, d_reward_):
        q_value = torch.Tensor(d_reward_).squeeze(-1)
        q_value = q_value[..., None]

        target_value = self.v(state_t1).squeeze(-1)
        target_value = target_value[..., None]
        critic_loss = self.c_loss(target_value, q_value)
        self.history_critic = critic_loss.detach().item()
        self.c_opt.zero_grad()
        critic_loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(self.v.parameters(), max_norm=1, norm_type=2)
        self.c_opt.step()

    def actor_update(self, state, action_acc, action_ori, advantage):
        with torch.autograd.detect_anomaly():
            action_ori = torch.FloatTensor(action_ori)
            action_acc = torch.FloatTensor(action_acc)
            action_cat = torch.stack((action_acc, action_ori), dim=1)

            pi_mult_m, pi_mult_s = self.pi(state)
            pi_mult_m_old, pi_mult_s_old = self.piold(state)

            if torch.any(torch.isnan(pi_mult_s)):
                print('invalid value sigma pi')

            if torch.any(torch.isnan(pi_mult_m)):
                print('invalid value mean pi')

            # 增加1e-8防止正态分布计算时除法越界
            pi_dist = Normal(pi_mult_m, pi_mult_s + 1e-8)
            pi_dist_old = Normal(pi_mult_m_old.detach(), pi_mult_s_old.detach() + 1e-8)

            logprob = pi_dist.cdf(action_cat)
            logprob_old = pi_dist_old.cdf(action_cat)

            ratio_acc = logprob[..., 0] / (logprob_old[..., 0] + 1e-8)
            ratio_ori = logprob[..., 1] / (logprob_old[..., 1] + 1e-8)

            if torch.any(torch.isnan(ratio_ori)) or torch.any(torch.isinf(ratio_ori)):
                print('invalid value sigma pi')
            if torch.any(torch.isnan(ratio_acc)) or torch.any(torch.isinf(ratio_acc)):
                print('invalid value sigma pi')

            # 切换ratio中inf值为固定值，防止inf进入backward计算
            # ratio_ori = torch.where(torch.isinf(ratio_ori), torch.full_like(ratio_ori, 3), ratio_ori)
            '''不确定混合acc和ori的mean和sigma梯度进行计算是否会造成梯度冲突'''
            # 使shape匹配，防止元素相乘发生广播问题
            ratio_acc, ratio_ori = ratio_acc[..., None], ratio_ori[..., None]
            assert ratio_acc.shape == advantage.shape
            surrogate1_acc = ratio_acc * advantage
            surrogate2_acc = torch.clamp(ratio_acc, 1-self.epilson, 1+self.epilson) * advantage

            assert ratio_ori.shape == advantage.shape
            surrogate1_ori = ratio_ori * advantage
            surrogate2_ori = torch.clamp(ratio_ori, 1-self.epilson, 1+self.epilson) * advantage

            actor_loss_acc = torch.min(torch.cat((surrogate1_acc, surrogate2_acc), dim=1), dim=1)[0]
            actor_loss_ori = torch.min(torch.cat((surrogate1_ori, surrogate2_ori), dim=1), dim=1)[0]

            self.a_opt.zero_grad()
            actor_loss = actor_loss_ori + actor_loss_acc
            actor_loss = -torch.mean(actor_loss)

            actor_loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(self.pi.parameters(), max_norm=1, norm_type=2)

            # print(self.pi.ori_meanDense4.weight.grad)

            self.a_opt.step()
            self.history_actor = actor_loss.detach().item()

    def update(self, state, action_acc, action_ori, discount_reward_):
        self.hard_update(self.pi, self.piold)
        state_ = torch.Tensor(state)
        act_acc = action_acc
        act_ori = action_ori
        d_reward = np.concatenate(discount_reward_).reshape(-1, 1)
        adv = self.advantage_calcu(d_reward, state_).detach()

        for i in range(self.update_actor_epoch):
            self.actor_update(state_, act_acc, act_ori, adv)
            # print(f'epochs: {self.ep}, time_steps: {self.t}, actor_loss: {self.history_actor}')

        for i in range(self.update_critic_epoch):
            self.critic_update(state_, d_reward)
            # print(f'epochs: {self.ep}, time_steps: {self.t}, critic_loss: {self.history_critic}')
        # log_dict = {'epochs': self.ep,
        #             'time_steps': {self.t},
        #             'actor_loss': {self.history_actor},
        #             'critic_loss': {self.history_critic}}

    def save_model(self, name):
        torch.save({'actor': self.pi.state_dict(),
                    'critic': self.v.state_dict(),
                    'opt_actor': self.a_opt.state_dict(),
                    'opt_critic': self.c_opt.state_dict()}, name)

    def load_model(self, name):
        checkpoints = torch.load(name)
        self.pi.load_state_dict(checkpoints['actor'])
        self.v.load_state_dict(checkpoints['critic'])
        self.a_opt.load_state_dict(checkpoints['opt_actor'])
        self.c_opt.load_state_dict(checkpoints['opt_critic'])

    @staticmethod
    def hard_update(model, target_model):
        weight_model = copy.deepcopy(model.state_dict())
        target_model.load_state_dict(weight_model)

    @staticmethod
    def data_pcs(obs_: dict):
        names = ['focus',
                 'speedX', 'speedY', 'speedZ',
                 'opponents',
                 'rpm',
                 'trackPos',
                 'wheelSpinVel',
                 'img',
                 'trackPos',
                 'angle']
        # for i in range(len(names)):
        #     exec('%s = obs_[i]' %names[i])
        focus_ = obs_[0]
        speedX_ = obs_[1]
        speedY_ = obs_[2]
        speedZ_ = obs_[3]
        opponent_ = obs_[4]
        rpm_ = obs_[5]
        track = obs_[6]
        wheelSpinel_ = obs_[7]
        img = obs_[8]
        trackPos = obs_[9]
        angle = obs_[10]
        img_data = np.zeros(shape=(64, 64, 3))
        for i in range(3):
            # img_data[:, :, i] = 255 - img[:, i].reshape((64, 64))
            img_data[:, :, i] = img[:, i].reshape((64, 64))
        img_data = Image.fromarray(img_data.astype(np.uint8))
        img_data = np.array(img_data.transpose(Image.FLIP_TOP_BOTTOM))
        img_data = rgb2gray(img_data).reshape(1, 1, img_data.shape[0], img_data.shape[1])
        return focus_, speedX_, speedY_, speedZ_, opponent_, rpm_, track, wheelSpinel_, img_data, trackPos, angle


if __name__ == '__main__':
    agent = PPO(8, 2, 16)
    obs = torch.randn((16, 8))
    agent.get_action(obs)
    agent.memory()