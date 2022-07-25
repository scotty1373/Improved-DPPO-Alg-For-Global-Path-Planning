# -*-  coding=utf-8 -*-
# @Time : 2022/5/16 9:57
# @Author : Scotty1373
# @File : utils.py
# @Software : PyCharm
import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from Envs.heatmap import normalize

TIME_BOUNDARY = 500
IMG_SIZE = (80, 80)
IMG_SIZE_RENDEER = 480


def trace_trans(vect, *, ratio=IMG_SIZE_RENDEER/16):
    remap_vect = np.array((vect[0] * ratio + (IMG_SIZE_RENDEER / 2), (-vect[1] * ratio) + IMG_SIZE_RENDEER), dtype=np.uint16)
    return remap_vect

# def heat_map_trans(vect, *, remap_sacle=REMAP_SACLE, ratio=REMAP_SACLE/ORG_SCALE):
#     remap_vect = np.array((vect[0] * ratio + remap_sacle/2, vect[1] * ratio), dtype=np.uint8)
#     return remap_vect


# 数据帧叠加
def state_frame_overlay(new_state, old_state, frame_num):
    new_frame_overlay = np.concatenate((new_state.reshape(1, -1),
                                        old_state.reshape(frame_num, -1)[:(frame_num - 1), ...]),
                                       axis=0).reshape(1, -1)
    return new_frame_overlay


# 基于图像的数据帧叠加
def pixel_based(new_state, old_state, frame_num):
    new_frame_overlay = np.concatenate((new_state,
                                       old_state[:, :(frame_num - 1), ...]),
                                       axis=1)
    return new_frame_overlay

# @profile
def img_proc(img, resize=(80, 80)):
    img = Image.fromarray(img.astype(np.uint8))
    img = np.array(img.resize(resize, resample=Image.BILINEAR))
    # img = img_ori.resize(resize, resample=Image.NEAREST)
    # img.show()
    # img = img_ori.resize(resize, resample=Image.BILINEAR)
    # img.show()
    # img = img_ori.resize(resize, Image.BICUBIC)
    # img.show()
    # img = img_ori.resize(resize, Image.ANTIALIAS)
    # img.show()
    img = rgb2gray(img).reshape(1, 1, 80, 80)
    img = normalize(img)
    return img


def record(global_ep, global_ep_r, ep_r, res_queue, worker_ep, name, idx):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put([idx, ep_r])

    print(f'{name}, '
          f'Global_EP: {global_ep.value}, '
          f'worker_EP: {worker_ep}, '
          f'EP_r: {global_ep_r.value}, '
          f'reward_ep: {ep_r}')

# @profile
def first_init(env, args):
    trace_history = []
    # 类装饰器不改变类内部调用方式
    obs, _, done, _ = env.reset()
    '''利用广播机制初始化state帧叠加结构，不使用stack重复对数组进行操作'''
    obs = (np.ones((args.frame_overlay, args.state_length)) * obs).reshape(1, -1)
    pixel_obs_ori = env.render(mode='rgb_array')
    pixel_obs = img_proc(pixel_obs_ori) * np.ones((1, args.frame_overlay, 80, 80))
    return trace_history, pixel_obs, obs, done


def cut_requires_grad(params):
    for param in params:
        param.requires_grad = False


def uniform_init(layer, *, a=-3e-3, b=3e-3):
    torch.nn.init.uniform_(layer.weight, a, b)
    torch.nn.init.constant_(layer.bias, 0)
    return layer


def orthogonal_init(layer, *, gain=1):
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    torch.nn.init.constant_(layer.bias, 0)
    return layer


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class ReplayBuffer:
    def __init__(self, max_lens, frame_overlay, state_length, action_dim, device):
        self.ptr = 0
        self.size = 0
        self.max_lens = max_lens
        self.state_length = state_length
        self.frame_overlay = frame_overlay
        self.action_dim = action_dim
        self.device = device
        self.rwd_rms = RunningMeanStd()
        self.pixel = np.zeros((self.max_lens, self.frame_overlay, 80, 80))
        self.next_pixel = np.zeros((self.max_lens, self.frame_overlay, 80, 80))
        self.vect = np.zeros((self.max_lens, self.state_length * self.frame_overlay))
        self.next_vect = np.zeros((self.max_lens, self.state_length * self.frame_overlay))
        self.reward = np.zeros((self.max_lens, 1))
        self.action = np.zeros((self.max_lens, self.action_dim))
        self.done = np.zeros((self.max_lens, 1))

    def add(self, pixel, next_pixel, vect, next_vect, reward, action, done):
        # reward rms update
        mean, std, count = reward.mean(), reward.std(), reward.shape[0]
        self.rwd_rms.update_from_moments(mean, std**2, count)

        self.pixel[self.ptr] = pixel.astype(np.float32)
        self.next_pixel[self.ptr] = next_pixel.astype(np.float32)
        self.vect[self.ptr] = vect.astype(np.float32)
        self.next_vect[self.ptr] = next_vect.astype(np.float32)
        self.reward[self.ptr] = reward.astype(np.float32)
        self.action[self.ptr] = action.astype(np.float32)
        self.done[self.ptr] = done.astype(np.float32)
        self.ptr = (self.ptr + 1) % self.max_lens
        self.size = min(self.size + 1, self.max_lens)

    def get_batch(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        # reward rms
        reward = self.reward[ind]
        reward = (reward - self.rwd_rms.mean) / (np.sqrt(self.rwd_rms.var) + 1e-4)

        return (torch.FloatTensor(self.pixel[ind]).to(self.device),
                torch.FloatTensor(self.next_pixel[ind]).to(self.device),
                torch.FloatTensor(self.vect[ind]).to(self.device),
                torch.FloatTensor(self.next_vect[ind]).to(self.device),
                torch.FloatTensor(reward).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.done[ind]).to(self.device))

