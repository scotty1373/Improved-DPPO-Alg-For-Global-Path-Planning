import numpy
import torch
from models.pixel_based import ActorModel, CriticModel
from torch.distributions import Normal
from collections import deque
from skimage.color import rgb2gray
from scipy import signal
from PIL import Image
import numpy as np
import copy

LEARNING_RATE_ACTOR = 0.5e-4
LEARNING_RATE_CRITIC = 2e-4
DECAY = 0.9
EPILSON = 0.2
torch.autograd.set_detect_anomaly(True)


class PPO:
    def __init__(self, state_dim, action_dim, batch_size, overlay, device):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.frame_overlay = overlay
        self.device = device

        # model build
        self._init(self.state_dim, self.action_dim, self.batch_size, self.frame_overlay, self.device)

        # optimizer initialize
        self.lr_actor = LEARNING_RATE_ACTOR
        self.lr_critic = LEARNING_RATE_CRITIC
        self.decay_index = DECAY
        self.epilson = EPILSON
        self.c_loss = torch.nn.MSELoss()
        self.clip_ratio = 0.2
        self.lamda = 0.95
        self.c_opt = torch.optim.Adam(params=self.v.parameters(), lr=self.lr_critic)
        self.a_opt = torch.optim.Adam(params=self.pi.parameters(), lr=self.lr_actor)
        self.c_sch = torch.optim.lr_scheduler.StepLR(self.c_opt, step_size=300, gamma=0.1)
        self.a_sch = torch.optim.lr_scheduler.StepLR(self.a_opt, step_size=300, gamma=0.1)

        # training configuration
        self.update_actor_epoch = 2
        self.update_critic_epoch = 2
        self.history_critic = 0
        self.history_actor = 0
        self.t = 0
        self.ep = 0

    def _init(self, state_dim, action_dim, train_batch, overlay, device):
        self.pi = ActorModel(state_dim, action_dim, overlay).to(device)
        self.v = CriticModel(state_dim, action_dim, overlay).to(device)
        self.memory = deque(maxlen=train_batch*2)

    def get_action(self, obs_):
        pixel_obs_, obs_ = torch.Tensor(copy.deepcopy(obs_[0])).to(self.device), torch.Tensor(copy.deepcopy(obs_[1])).to(self.device)

        self.pi.eval()
        with torch.no_grad():
            action, action_logprob, dist = self.pi(pixel_obs_, obs_)
        self.pi.train()

        return action.cpu().detach().numpy(), action_logprob.cpu().detach().numpy(), dist

    def state_store_memory(self, pixel_s, s, act, r, logprob):
        self.memory.append((pixel_s, s, act, r, logprob))

    # 计算reward衰减，根据马尔可夫过程，从最后一个reward向前推
    def decayed_reward(self, singal_state_frame, reward_):
        decayed_rd = []
        with torch.no_grad():
            state_frame_pixel = torch.Tensor(singal_state_frame[0]).to(self.device)
            state_frame_vect = torch.Tensor(singal_state_frame[1]).to(self.device)
            value_target = self.v(state_frame_pixel, state_frame_vect).cpu().detach().numpy()
            for rd_ in reward_[::-1]:
                value_target = rd_ + value_target * self.decay_index
                decayed_rd.append(value_target)
            decayed_rd.reverse()
        return decayed_rd

    # 计算actor更新用的advantage value
    def advantage_calcu(self, decay_reward, state_t):
        with torch.no_grad():
            state_t = torch.Tensor(state_t).to(self.device)
            critic_value_ = self.v(state_t)
            d_reward = torch.Tensor(decay_reward)
            advantage = d_reward - critic_value_
        # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        if torch.isnan(advantage).any():
            print("advantage is nan")
        return advantage

    def gae_adv(self, state_pixel, state_vect, reward_step, last_val):
        with torch.no_grad():
            critic_value_ = self.v(state_pixel, state_vect)
            critic_value_ = torch.cat([critic_value_, last_val.reshape(-1, 1)], dim=0)

            assert reward_step.shape == critic_value_.shape
            td_error = reward_step[:-1, ...] + self.decay_index * critic_value_[1:, ...] - critic_value_[:-1, ...]
            td_error = td_error.cpu().numpy()

            gae_advantage = signal.lfilter([1], [1, -self.decay_index*self.lamda], td_error[::-1, ...], axis=0)[::-1, ...]

            # for idx, td in enumerate(td_error.numpy()[::-1]):
            #     temp = 0
            #     for adv_idx, weight in enumerate(range(idx, -1, -1)):
            #         temp += gae_advantage[adv_idx, 0] * ((self.lamda*self.epilson)**weight)
            #     gae_advantage[idx, ...] = td + temp
            """！！！以下代码结构需做优化！！！"""
            gae_advantage = torch.Tensor(gae_advantage.copy()).to(self.device)
        # gae_advantage = (gae_advantage - gae_advantage.mean()) / (gae_advantage.std() + 1e-8)
        return gae_advantage

    # 计算critic更新用的 Q(s, a)和 V(s)
    def critic_update(self, pixel_state, vect_state, d_reward_):
        q_value = torch.Tensor(d_reward_).squeeze(-1).to(self.device)
        q_value = q_value[..., None]

        target_value = self.v(pixel_state, vect_state).squeeze(-1)
        target_value = target_value[..., None]
        assert target_value.shape == q_value.shape
        critic_loss = self.c_loss(target_value, q_value)
        self.history_critic = critic_loss.detach().item()
        self.c_opt.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.v.parameters(), max_norm=2000, norm_type=2)
        self.c_opt.step()

    def actor_update(self, pixel_state, vect_state, action, logprob_old, advantage):
        _, _, pi_dist = self.pi(pixel_state, vect_state)
        logprob = pi_dist.log_prob(action)

        pi_entropy = pi_dist.entropy().mean(dim=1).detach()

        """是否需要增加kl散度监视？？？"""

        assert logprob.shape == logprob_old.shape
        ratio = torch.exp(torch.sum(logprob - logprob_old, dim=-1))

        # 使shape匹配，防止元素相乘发生广播问题
        ratio = torch.unsqueeze(ratio, dim=1)
        assert ratio.shape == advantage.shape
        surrogate1_acc = ratio * advantage
        surrogate2_acc = torch.clamp(ratio, 1-self.epilson, 1+self.epilson) * advantage

        actor_loss = torch.min(torch.cat((surrogate1_acc, surrogate2_acc), dim=1), dim=1)[0]

        self.a_opt.zero_grad()
        actor_loss = -torch.mean(actor_loss)

        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.pi.parameters(), max_norm=0.5, norm_type=2)

        self.a_opt.step()
        self.history_actor = actor_loss.detach().item()

    # 用于获取单幕中用于更新actor和critic的advantage和reward_sum
    def get_trjt(self, last_pixel, last_vect, done):
        pixel_state, vect_state, action, reward_nstep, logprob_nstep = zip(*self.memory)
        pixel_state = np.concatenate(pixel_state, axis=0)
        vect_state = np.concatenate(vect_state, axis=0)
        action = np.concatenate(action, axis=0)
        reward_nstep = np.stack(reward_nstep, axis=0)
        logprob_nstep = np.concatenate(logprob_nstep, axis=0)
        # 动作价值计算
        discount_reward = self.decayed_reward((last_pixel, last_vect), reward_nstep)
        with torch.no_grad():
            last_frame_pixel = torch.Tensor(last_pixel).to(self.device)
            last_frame_vect = torch.Tensor(last_vect).to(self.device)
            last_val = self.v(last_frame_pixel, last_frame_vect)

        pixel_state = torch.Tensor(pixel_state).to(self.device)
        vect_state = torch.Tensor(vect_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        logprob_nstep = torch.FloatTensor(logprob_nstep).to(self.device)
        reward_nstep = reward_nstep.reshape(-1, 1)
        if done:
            last_val = torch.zeros((1, 1)).to(self.device)
        try:
            reward = torch.FloatTensor(reward_nstep).to(self.device)
            reward = torch.cat((reward, last_val), dim=0)
        except TypeError as e:
            print('reward error')

        # 用于计算critic损失/原始advantage
        d_reward = np.concatenate(discount_reward).reshape(-1, 1)

        adv = self.gae_adv(pixel_state, vect_state, reward, last_val)
        return pixel_state, vect_state, action, logprob_nstep, d_reward, adv

    def update(self, pixel_state, vect_state, action, logprob, d_reward, adv):
        for i in range(self.update_actor_epoch):
            self.actor_update(pixel_state, vect_state, action, logprob, adv)
            # print(f'epochs: {self.ep}, time_steps: {self.t}, actor_loss: {self.history_actor}')

        for i in range(self.update_critic_epoch):
            self.critic_update(pixel_state, vect_state, d_reward)
            # print(f'epochs: {self.ep}, time_steps: {self.t}, critic_loss: {self.history_critic}')

    def save_model(self, name):
        torch.save({'actor': self.pi.state_dict(),
                    'critic': self.v.state_dict(),
                    'opt_actor': self.a_opt.state_dict(),
                    'opt_critic': self.c_opt.state_dict()}, name)

    def load_model(self, name):
        checkpoints = torch.load(name)
        self.pi.load_state_dict(checkpoints['actor'], strict=False)
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