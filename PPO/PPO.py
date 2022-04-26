import numpy
import torch
from models.net_builder import ActorModel, CriticModel
from torch.distributions import Normal
from collections import deque
from skimage.color import rgb2gray
from PIL import Image
import numpy as np
import copy

LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 5e-4
DECAY = 0.99
EPILSON = 0.15
# torch.autograd.set_detect_anomaly(True)


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
        self.clip_ratio = 0.2
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
        # self.piold = ActorModel(state_dim, action_dim)
        self.v = CriticModel(state_dim, action_dim)
        self.memory = deque(maxlen=train_batch)

    def get_action(self, obs_):
        obs_ = torch.Tensor(copy.deepcopy(obs_))
        self.pi.eval()
        with torch.no_grad():
            action, action_logprob, dist = self.pi(obs_)
        self.pi.train()

        return action.cpu().detach().numpy(), action_logprob.detach().numpy(), dist

    def state_store_memory(self, s, act, r, logprob):
        self.memory.append((s, act, r, logprob))

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
        torch.nn.utils.clip_grad_norm_(self.v.parameters(), max_norm=0.5, norm_type=2)
        self.c_opt.step()

    def actor_update(self, state, action, logprob_old, advantage):
        action = torch.FloatTensor(action)
        logprob_old = torch.FloatTensor(logprob_old)

        _, _, pi_dist = self.pi(state)
        logprob = pi_dist.log_prob(action)
        """取消logprob进行sum操作后logprob_old维度缺失问题"""
        # logprob_old = torch.FloatTensor(logprob_old).unsqueeze(-1)
        #
        # _, _, pi_dist = self.pi(state)
        # logprob = pi_dist.log_prob(action).sum(-1).unsqueeze(-1)

        pi_entropy = pi_dist.entropy().mean(dim=1).detach()

        assert logprob.shape == logprob_old.shape
        ratio = torch.exp(torch.sum(logprob - logprob_old, dim=-1))

        # 切换ratio中inf值为固定值，防止inf进入backward计算
        # ratio_ori = torch.where(torch.isinf(ratio_ori), torch.full_like(ratio_ori, 3), ratio_ori)

        # 使shape匹配，防止元素相乘发生广播问题
        ratio = torch.unsqueeze(ratio, dim=1)
        assert ratio.shape == advantage.shape
        surrogate1_acc = ratio * advantage
        surrogate2_acc = torch.clamp(ratio, 1-self.epilson, 1+self.epilson) * advantage

        actor_loss = torch.min(torch.cat((surrogate1_acc, surrogate2_acc), dim=1), dim=1)[0] + pi_entropy

        self.a_opt.zero_grad()
        actor_loss = -torch.mean(actor_loss)

        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.pi.parameters(), max_norm=0.5, norm_type=2)

        self.a_opt.step()
        self.history_actor = actor_loss.detach().item()

    def update(self, state, action, logprob, discount_reward_):
        state_ = torch.Tensor(state)
        act = action
        d_reward = np.concatenate(discount_reward_).reshape(-1, 1)
        adv = self.advantage_calcu(d_reward, state_)

        for i in range(self.update_actor_epoch):
            self.actor_update(state_, act, logprob, adv)
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