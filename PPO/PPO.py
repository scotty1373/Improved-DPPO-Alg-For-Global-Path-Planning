import numpy
import torch
from models.pixel_based import ActorModel, CriticModel
from utils_tools.utils import RunningMeanStd
from RND.rnd_model import RNDModel
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
        self.d_rwd_ext = []
        self.d_rwd_int = []
        self.adv = []
        self.next_state = []

    def collect_trajorbatch(self, pixel, vect, act, logp, d_rwd_ext, d_rwd_int, adv, next_state):
        self.pixel_state.append(pixel)
        self.vect_state.append(vect)
        self.action.append(act)
        self.logprob.append(logp)
        self.d_rwd_ext.append(d_rwd_ext)
        self.d_rwd_int.append(d_rwd_int)
        self.adv.append(adv)
        self.next_state.append(next_state)

    def collect_batch(self, pixel, vect, act, logp, d_rwd_ext, d_rwd_int, adv, next_state):
        self.pixel_state += pixel
        self.vect_state += vect
        self.action += act
        self.logprob += logp
        self.d_rwd_ext += d_rwd_ext
        self.d_rwd_int += d_rwd_int
        self.adv += adv
        self.next_state += next_state

    def get_data(self, device):
        self.pixel_state = torch.cat(self.pixel_state, dim=0).to(device)
        self.vect_state = torch.cat(self.vect_state, dim=0).to(device)
        self.action = torch.cat(self.action, dim=0).to(device)
        self.logprob = torch.cat(self.logprob, dim=0).to(device)
        self.d_rwd_ext = torch.cat(self.d_rwd_ext, dim=0).to(device)
        self.d_rwd_int = torch.cat(self.d_rwd_int, dim=0).to(device)
        self.adv = torch.cat(self.adv, dim=0).to(device)
        self.next_state = (np.concatenate([pixel[0] for pixel in self.next_state], axis=0),
                           np.concatenate([pixel[1] for pixel in self.next_state], axis=0))
        self.next_state = (torch.FloatTensor(self.next_state[0]).to(device),
                           torch.FloatTensor(self.next_state[1]).to(device))

    def cleanup(self):
        self.pixel_state = []
        self.vect_state = []
        self.action = []
        self.logprob = []
        self.d_rwd_ext = []
        self.d_rwd_int = []
        self.adv = []
        self.next_state = []


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
        return pixel, obs, reward, done, info

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
        self.reward_dc_ext = 0.999
        self.reward_dc_int = 0.99
        self.epilson = EPILSON
        self.c_loss = torch.nn.MSELoss()
        self.lamda = 0.95
        self.kl_target = 0.01
        self.c_opt = torch.optim.Adam(params=self.v.parameters(), lr=self.lr_critic)
        self.a_opt = torch.optim.Adam(params=self.pi.parameters(), lr=self.lr_actor)
        self.c_sch = torch.optim.lr_scheduler.StepLR(self.c_opt, step_size=500, gamma=0.1)
        self.a_sch = torch.optim.lr_scheduler.StepLR(self.a_opt, step_size=500, gamma=0.1)

        # training configuration
        self.history_critic = 0
        self.history_actor = 0
        self.t = 0
        self.ep = 0

        # RND block
        self.extcoef = 1.0
        self.intcoef = 0.5
        self.update_proportion = 0.25
        self.rnd_loss_func = torch.nn.MSELoss(reduction='none')
        self.rnd_opt = torch.optim.Adam(self.rnd.predict_structure.parameters(), lr=0.0001)
        self.staterms = RunningMeanStd(shape=(1, 1, 80, 80))
        self.vectrms = RunningMeanStd(shape=(1, self.state_length))
        # self.overlay_state_rms = RunningMeanStd(shape=(1, self.frame_overlay, 80, 80))
        # self.overlay_vect_rms = RunningMeanStd(shape=(1, self.state_length*self.frame_overlay))
        self.rwd_int_rms = RunningMeanStd()
        self.rwd_ext_rms = RunningMeanStd()

    def _init(self, state_dim, action_dim, overlay, device):
        self.pi = ActorModel(state_dim, action_dim, overlay).to(device)
        self.v = CriticModel(state_dim, action_dim, overlay).to(device)
        self.memory = deque(maxlen=max_mem_len)
        self.rnd = RNDModel(self.state_length).to(device)

    def get_action(self, obs_):
        # pixel_obs_ = (obs_[0] - self.overlay_state_rms.mean) / np.sqrt(self.overlay_state_rms.var)
        # obs_ = (obs_[1] - self.overlay_vect_rms.mean) / np.sqrt(self.overlay_vect_rms.var)
        pixel_obs_, obs_ = torch.Tensor(copy.deepcopy(obs_[0])).to(self.device), torch.Tensor(copy.deepcopy(obs_[1])).to(self.device)

        self.pi.eval()
        with torch.no_grad():
            action, action_logprob, dist = self.pi(pixel_obs_, obs_)
        self.pi.train()

        return action.cpu().detach().numpy(), action_logprob.cpu().detach().numpy(), dist

    def state_store_memory(self, pixel_s, s, act, r, logprob, next_pixel_s, next_s):
        self.memory.append((pixel_s, s, act, r, logprob, next_pixel_s, next_s))

    # 计算reward衰减，根据马尔可夫过程，从最后一个reward向前推
    def decayed_reward(self, singal_state_frame, reward_ext, reward_int):
        decay_rd_ext = []
        decay_rd_int = []
        with torch.no_grad():
            state_frame_pixel = torch.Tensor(singal_state_frame[0]).to(self.device)
            state_frame_vect = torch.Tensor(singal_state_frame[1]).to(self.device)
            value_ext, value_int = self.v(state_frame_pixel, state_frame_vect)
            value_ext = value_ext.cpu().detach().numpy()
            value_int = value_int.cpu().detach().numpy()
            for rd_ext, rd_int in zip(reward_ext[::-1], reward_int[::-1]):
                value_ext = rd_ext + value_ext * self.reward_dc_ext
                value_int = rd_int + value_int * self.reward_dc_int
                decay_rd_ext.append(value_ext)
                decay_rd_int.append(value_int)
            decay_rd_ext.reverse()
            decay_rd_int.reverse()
        return decay_rd_ext, decay_rd_int

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

    def gae_adv(self, state_pixel, state_vect, rwd_ext, rwd_int, last_val_ext, last_val_int):
        with torch.no_grad():
            value_ext, value_int = self.v(state_pixel, state_vect)
            value_ext = torch.cat([value_ext, last_val_ext.reshape(-1, 1)], dim=0)
            value_int = torch.cat([value_int, last_val_int.reshape(-1, 1)], dim=0)

            assert rwd_ext.shape == value_ext.shape
            assert rwd_int.shape == value_int.shape
            # 计算内部奖励advantage
            td_error_ext = rwd_ext[:-1, ...] + self.reward_dc_ext * value_ext[1:, ...] - value_ext[:-1, ...]
            td_error_ext = td_error_ext.cpu().numpy()
            # 计算外部奖励advantage
            td_error_int = rwd_int[:-1, ...] + self.reward_dc_int * value_int[1:, ...] - value_int[:-1, ...]
            td_error_int = td_error_int.cpu().numpy()

            # 计算内部奖励gae
            gae_ext = signal.lfilter([1], [1, -self.reward_dc_ext*self.lamda], td_error_ext[::-1, ...], axis=0)[::-1, ...]
            # 计算外部奖励gae
            gae_int = signal.lfilter([1], [1, -self.reward_dc_int*self.lamda], td_error_int[::-1, ...], axis=0)[::-1, ...]

            # [todo]！！！以下代码结构需做优化！！！
            gae_ext = torch.Tensor(gae_ext.copy()).to(self.device)
            gae_int = torch.Tensor(gae_int.copy()).to(self.device)
        return gae_ext, gae_int

    # 计算critic更新用的 Q(s, a)和 V(s)
    def critic_update(self, pixel_state, vect_state, d_rwd_ext, d_rwd_int):
        q_val_ext = d_rwd_ext.squeeze(-1).to(self.device)
        q_val_int = d_rwd_int.squeeze(-1).to(self.device)
        q_val_ext = q_val_ext[..., None]
        q_val_int = q_val_int[..., None]

        val_ext, val_int = self.v(pixel_state, vect_state)
        val_ext = val_ext.squeeze(-1)[..., None]
        val_int = val_int.squeeze(-1)[..., None]
        self.c_opt.zero_grad()

        assert q_val_ext.shape == val_ext.shape
        assert q_val_int.shape == val_int.shape
        critic_ext_loss = self.c_loss(val_ext, q_val_ext)
        critic_int_loss = self.c_loss(val_int, q_val_int)
        critic_loss = critic_ext_loss + critic_int_loss
        self.history_critic = critic_loss.cpu().detach().item()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.v.parameters(), clip_value=100)
        self.c_opt.step()

    def actor_update(self, pixel_state, vect_state, action, logprob_old, advantage):
        _, _, pi_dist = self.pi(pixel_state, vect_state)
        logprob = pi_dist.log_prob(action)

        pi_entropy = pi_dist.entropy().mean(dim=1)

        # [todo] 需要增加KL散度计算
        beta = 1
        kl_div = torch.nn.functional.kl_div(logprob, logprob_old, reduction='mean')
        if kl_div >= 1.5 * self.kl_target:
            beta = beta * 2
        elif kl_div < self.kl_target / 1.5:
            beta = beta / 2

        assert logprob.shape == logprob_old.shape
        ratio = torch.exp(torch.sum(logprob - logprob_old, dim=-1))

        # 使shape匹配，防止元素相乘发生广播问题
        ratio = torch.unsqueeze(ratio, dim=1)
        assert ratio.shape == advantage.shape
        advantage = (advantage - advantage.mean()) / advantage.std()
        surrogate1_acc = ratio * advantage
        surrogate2_acc = torch.clamp(ratio, 1-self.epilson, 1+self.epilson) * advantage

        actor_loss = torch.min(torch.cat((surrogate1_acc, surrogate2_acc), dim=1), dim=1)[0] - kl_div * beta

        self.a_opt.zero_grad()
        actor_loss = -torch.mean(actor_loss)
        try:
            actor_loss.backward()
        except RuntimeError as e:
            print('Exp backward detected!!!')
        torch.nn.utils.clip_grad_norm_(self.pi.parameters(), max_norm=1, norm_type=2)

        self.a_opt.step()
        self.history_actor = actor_loss.detach().item()

    def rnd_update(self, pixel, vect):
        target_vect, predict_vect = self.rnd(pixel, vect)
        target_vect = target_vect.detach()
        self.rnd_opt.zero_grad()
        loss = self.rnd_loss_func(predict_vect, target_vect).mean(-1)
        # Proportion of exp used for predictor update
        """from the RND pytorch complete code"""
        """https://github.com/jcwleo/random-network-distillation-pytorch/blob/e383fb95177c50bfdcd81b43e37c443c8cde1d94/agents.py"""
        mask = torch.rand(len(loss)).to(self.device)
        mask = (mask < self.update_proportion).to(self.device)
        loss = (loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
        loss.backward()
        # 未做梯度裁剪
        self.rnd_opt.step()
        return loss.cpu().detach().numpy()

    def calculate_intrinsic_reward(self, pixel_state, vect_state):
        pixel_state = torch.FloatTensor(pixel_state).to(self.device)
        vect_state = torch.FloatTensor(vect_state).to(self.device)
        self.rnd.eval()
        target_feature, predict_feature = self.rnd(pixel_state, vect_state)
        intrinsic_reward = (target_feature - predict_feature).pow(2).sum(-1) / 2
        self.rnd.train()
        return intrinsic_reward.cpu().detach().numpy()

    # 用于获取单幕中用于更新actor和critic的advantage和reward_sum
    def get_trjt(self, last_pixel, last_vect, done):
        pixel_state, vect_state, action, reward_ext, logprob_nstep, next_pixel_state, next_vect_state = zip(*self.memory)
        pixel_state = np.concatenate(pixel_state, axis=0)
        vect_state = np.concatenate(vect_state, axis=0)
        action = np.concatenate(action, axis=0)
        reward_ext = np.stack(reward_ext, axis=0)
        logprob_nstep = np.concatenate(logprob_nstep, axis=0)
        next_pixel_state = np.concatenate(next_pixel_state, axis=0)
        next_vect_state = np.concatenate(next_vect_state, axis=0)

        # state vect rms
        # self.overlay_state_rms.update(pixel_state)
        # self.overlay_vect_rms.update(vect_state)
        # pixel_state = (pixel_state - self.overlay_state_rms.mean) / np.sqrt(self.overlay_state_rms.var)
        # vect_state = (vect_state - self.overlay_vect_rms.mean) / np.sqrt(self.overlay_vect_rms.var)

        # tensor 类型转换
        pixel_state = torch.Tensor(pixel_state)
        vect_state = torch.Tensor(vect_state)
        action = torch.FloatTensor(action)
        logprob_nstep = torch.FloatTensor(logprob_nstep)

        # intrinsic reward计算
        self.staterms.update(next_pixel_state)
        self.vectrms.update(next_vect_state)
        next_pixel_state = np.clip(((next_pixel_state - self.staterms.mean) / np.sqrt(self.staterms.var)), -5, 5)
        next_vect_state = np.clip(((next_vect_state - self.vectrms.mean) / np.sqrt(self.vectrms.var)), -5, 5)
        intrinsic_reward = self.calculate_intrinsic_reward(next_pixel_state, next_vect_state)
        mean, std, count = intrinsic_reward.mean(), intrinsic_reward.std(), intrinsic_reward.shape[0]
        self.rwd_int_rms.update_from_moments(mean, std ** 2, count)
        intrinsic_reward = (intrinsic_reward - self.rwd_int_rms.mean) / np.sqrt(self.rwd_int_rms.var)

        # ext reward计算
        mean, std, count = reward_ext.mean(), reward_ext.std(), reward_ext.shape[0]
        self.rwd_ext_rms.update_from_moments(mean, std ** 2, count)
        reward_ext = (reward_ext - self.rwd_ext_rms.mean) / np.sqrt(self.rwd_ext_rms.var)

        # discount reward计算
        discount_rd_ext, discount_rd_int = self.decayed_reward((last_pixel, last_vect), reward_ext, intrinsic_reward)

        # 计算最后一个状态的内在回报和外在回报
        if done:
            last_val_ext = last_val_int = torch.zeros((1, 1))
        else:
            with torch.no_grad():
                last_frame_pixel = torch.Tensor(last_pixel)
                last_frame_vect = torch.Tensor(last_vect)
                last_val_ext, last_val_int = self.v(last_frame_pixel, last_frame_vect)

        try:
            reward_ext = torch.FloatTensor(reward_ext).reshape(-1, 1)
            reward_int = torch.FloatTensor(intrinsic_reward).reshape(-1, 1)
            reward_ext = torch.cat((reward_ext, last_val_ext), dim=0)
            reward_int = torch.cat((reward_int, last_val_int), dim=0)
        except TypeError as e:
            print('reward error')

        # 用于计算critic损失/原始advantage
        d_rwd_ext = torch.Tensor(np.concatenate(discount_rd_ext).reshape(-1, 1))
        d_rwd_int = torch.Tensor(np.concatenate(discount_rd_int).reshape(-1, 1))

        gae_ext, gae_int = self.gae_adv(state_pixel=pixel_state, state_vect=vect_state,
                                        rwd_ext=reward_ext, rwd_int=reward_int,
                                        last_val_ext=last_val_ext, last_val_int=last_val_int)
        gae = gae_ext * self.extcoef + gae_int * self.intcoef
        return pixel_state, vect_state, action, logprob_nstep, d_rwd_ext, d_rwd_int, gae, (next_pixel_state, next_vect_state), intrinsic_reward

    def update(self, buffer, args):
        buffer.get_data(self.device)
        indice = torch.randperm(args.max_timestep * args.worker_num).to(self.device)
        iter_times = args.max_timestep * args.worker_num//self.batch_size
        for i in range(args.max_timestep * args.worker_num//self.batch_size):
            try:
                batch_index = indice[i*self.batch_size:(i+1)*self.batch_size]
                batch_pixel = torch.index_select(buffer.pixel_state, dim=0, index=batch_index)
                batch_vect = torch.index_select(buffer.vect_state, dim=0, index=batch_index)
                batch_action = torch.index_select(buffer.action, dim=0, index=batch_index)
                batch_logprob = torch.index_select(buffer.logprob, dim=0, index=batch_index)
                batch_d_rwd_ext = torch.index_select(buffer.d_rwd_ext, dim=0, index=batch_index)
                batch_d_rwd_int = torch.index_select(buffer.d_rwd_int, dim=0, index=batch_index)
                batch_adv = torch.index_select(buffer.adv, dim=0, index=batch_index)
                batch_next_pixel = torch.index_select(buffer.next_state[0], dim=0, index=batch_index)
                batch_next_vect = torch.index_select(buffer.next_state[1], dim=0, index=batch_index)
            except IndexError as e:
                print('index error')
            self.actor_update(batch_pixel, batch_vect, batch_action, batch_logprob, batch_adv)
            self.critic_update(batch_pixel, batch_vect, batch_d_rwd_ext, batch_d_rwd_int)
            loss_rnd = self.rnd_update(batch_next_pixel, batch_next_vect)
            self.logger.add_scalar(tag='Loss/actor_loss',
                                   scalar_value=self.history_actor,
                                   global_step=self.ep * iter_times + i)
            self.logger.add_scalar(tag='Loss/critic_loss',
                                   scalar_value=self.history_critic,
                                   global_step=self.ep * iter_times + i)
            self.logger.add_scalar(tag='Loss/rnd_loss',
                                   scalar_value=loss_rnd,
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