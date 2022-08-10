import numpy
import torch
from models.pixel_based import ActorModel, CriticModel
from utils_tools.utils import RunningMeanStd
from torch.distributions import Normal
import gym
from collections import deque
from skimage.color import rgb2gray
from scipy import signal
from PIL import Image
import numpy as np
import copy

LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 5e-5
DECAY = 0.98
EPILSON = 0.1
max_mem_len = 512


class PPO_Buffer:
    def __init__(self):
        self.pixel_state = []
        self.vect_state = []
        self.action = []
        self.logprob = []
        self.d_reward = []
        self.adv = []

    def collect_trajorbatch(self, pixel, vect, act, logp, d_reward, adv):
        self.pixel_state.append(pixel)
        self.vect_state.append(vect)
        self.action.append(act)
        self.logprob.append(logp)
        self.d_reward.append(d_reward)
        self.adv.append(adv)

    def collect_batch(self, pixel_state, vect_state, action, logprob, d_reward, adv):
        self.pixel_state += pixel_state
        self.vect_state += vect_state
        self.action += action
        self.logprob += logprob
        self.d_reward += d_reward
        self.adv += adv

    def get_data(self, device):
        self.pixel_state = torch.cat(self.pixel_state, dim=0).to(device)
        self.vect_state = torch.cat(self.vect_state, dim=0).to(device)
        self.action = torch.cat(self.action, dim=0).to(device)
        self.logprob = torch.cat(self.logprob, dim=0).to(device)
        self.d_reward = torch.cat(self.d_reward, dim=0).to(device)
        self.adv = torch.cat(self.adv, dim=0).to(device)

    def cleanup(self):
        self.pixel_state = []
        self.vect_state = []
        self.action = []
        self.logprob = []
        self.d_reward = []
        self.adv = []


class SkipEnvFrame(gym.Wrapper):
    def __init__(self, env, skip=3):
        super(SkipEnvFrame, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            pixel = self.env.render()
            total_reward += reward
            if done:
                break
        return pixel, obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class PPO:
    def __init__(self, frame_overlay, state_length, action_dim, batch_size, overlay, device, logger=None, root=True):
        self.state_length = state_length
        self.state_dim = frame_overlay * state_length
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.frame_overlay = overlay
        self.device = device
        self.logger = logger

        # [todo] 父进程标识符
        self.root = root

        # model build
        self._init(self.state_dim, self.action_dim, self.frame_overlay, self.device)

        # optimizer initialize
        self.lr_actor = LEARNING_RATE_ACTOR
        self.lr_critic = LEARNING_RATE_CRITIC
        self.decay_index = DECAY
        self.epsilon = EPILSON
        self.c_loss = torch.nn.MSELoss()
        self.lamda = 0.95
        self.kl_target = 0.01
        self.c_opt = torch.optim.Adam(params=self.v.parameters(), lr=self.lr_critic)
        self.a_opt = torch.optim.Adam(params=self.pi.parameters(), lr=self.lr_actor)
        self.c_sch = torch.optim.lr_scheduler.StepLR(self.c_opt, step_size=500, gamma=0.1)
        self.a_sch = torch.optim.lr_scheduler.StepLR(self.a_opt, step_size=500, gamma=0.1)

        # training configuration
        self.state_rms = RunningMeanStd(shape=(1, self.frame_overlay, 80, 80))
        self.vect_rms = RunningMeanStd(shape=(1, self.frame_overlay*self.state_length))
        self.reward_rms = RunningMeanStd()
        self.history_critic = 0
        self.history_actor = 0
        self.t = 0
        self.ep = 0

    def _init(self, state_dim, action_dim, overlay, device):
        self.pi = ActorModel(state_dim, action_dim, overlay).to(device)
        self.v = CriticModel(state_dim, action_dim, overlay).to(device)
        self.memory = deque(maxlen=max_mem_len)

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
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-4)
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
        q_value = d_reward_.squeeze(-1).to(self.device)
        q_value = q_value[..., None]

        target_value = self.v(pixel_state, vect_state).squeeze(-1)
        target_value = target_value[..., None]
        self.c_opt.zero_grad()
        assert target_value.shape == q_value.shape
        critic_loss = self.c_loss(target_value, q_value)
        self.history_critic = critic_loss.detach().item()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.v.parameters(), clip_value=100)
        self.c_opt.step()

    def actor_update(self, pixel_state, vect_state, action, logprob_old, advantage):
        _, _, pi_dist = self.pi(pixel_state, vect_state)
        logprob = pi_dist.log_prob(action)

        pi_entropy = pi_dist.entropy().mean()

        assert logprob.shape == logprob_old.shape
        ratio = torch.exp(torch.sum(logprob - logprob_old, dim=-1))

        # 使shape匹配，防止元素相乘发生广播问题
        ratio = torch.unsqueeze(ratio, dim=1)
        assert ratio.shape == advantage.shape
        advantage = (advantage - advantage.mean()) / advantage.std()
        surrogate1_acc = ratio * advantage
        surrogate2_acc = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantage

        # [todo] 需要增加KL散度计算
        beta = 1
        kl_approx_div = torch.nn.functional.kl_div(logprob, logprob_old, reduction='mean')
        # kl_approx_div = ((ratio - 1) - (logprob - logprob_old)).mean()
        if kl_approx_div >= 1.5 * self.kl_target:
            beta = beta * 2
        elif kl_approx_div < self.kl_target / 1.5:
            beta = beta / 2

        actor_loss = torch.min(torch.cat((surrogate1_acc, surrogate2_acc), dim=1), dim=1)[0]

        self.a_opt.zero_grad()
        actor_loss = -torch.mean(actor_loss) + kl_approx_div * beta - (pi_entropy * 0.001)
        try:
            actor_loss.backward()
        except RuntimeError as e:
            print('Exp backward detected!!!')
        torch.nn.utils.clip_grad_norm_(self.pi.parameters(), max_norm=1, norm_type=2)

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

        pixel_state = torch.Tensor(pixel_state)
        vect_state = torch.Tensor(vect_state)
        action = torch.FloatTensor(action)
        logprob_nstep = torch.FloatTensor(logprob_nstep)

        # reward rms
        reward_nstep = reward_nstep.reshape(-1, 1)
        mean, std, count = reward_nstep.mean(), reward_nstep.std(), reward_nstep.shape[0]
        self.reward_rms.update_from_moments(mean, std**2, count)
        reward_nstep = (reward_nstep - self.reward_rms.mean) / np.sqrt(self.reward_rms.var)

        # 动作价值计算
        discount_reward = self.decayed_reward((last_pixel, last_vect), reward_nstep)
        with torch.no_grad():
            last_frame_pixel = torch.Tensor(last_pixel)
            last_frame_vect = torch.Tensor(last_vect)
            last_val = self.v(last_frame_pixel, last_frame_vect)

        if done:
            last_val = torch.zeros((1, 1))
        try:
            reward = torch.FloatTensor(reward_nstep)
            reward = torch.cat((reward, last_val), dim=0)
        except TypeError as e:
            print('reward error')

        # 用于计算critic损失/原始advantage
        d_reward = torch.Tensor(np.concatenate(discount_reward).reshape(-1, 1))

        adv = self.gae_adv(pixel_state, vect_state, reward, last_val)
        return pixel_state, vect_state, action, logprob_nstep, d_reward, adv

    def update(self, buffer, args):
        buffer.get_data(self.device)
        indices = torch.randperm(args.max_timestep * args.worker_num).to(self.device)
        iter_times = args.max_timestep * args.worker_num//self.batch_size
        for i in range(args.max_timestep * args.worker_num//self.batch_size):
            try:
                batch_index = indices[i*self.batch_size:(i+1)*self.batch_size]
                batch_pixel = torch.index_select(buffer.pixel_state, dim=0, index=batch_index)
                batch_vect = torch.index_select(buffer.vect_state, dim=0, index=batch_index)
                batch_action = torch.index_select(buffer.action, dim=0, index=batch_index)
                batch_d_reward = torch.index_select(buffer.d_reward, dim=0, index=batch_index)
                batch_logprob = torch.index_select(buffer.logprob, dim=0, index=batch_index)
                batch_adv = torch.index_select(buffer.adv, dim=0, index=batch_index)
            except IndexError as e:
                print('get trajectory index error')
            self.actor_update(batch_pixel, batch_vect, batch_action, batch_logprob, batch_adv)
            self.critic_update(batch_pixel, batch_vect, batch_d_reward)
            self.logger.add_scalar(tag='Loss/actor_loss',
                                   scalar_value=self.history_actor,
                                   global_step=self.ep * iter_times + i)
            self.logger.add_scalar(tag='Loss/critic_loss',
                                   scalar_value=self.history_critic,
                                   global_step=self.ep * iter_times + i)
        buffer.cleanup()

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
