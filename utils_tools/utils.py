# -*-  coding=utf-8 -*-
# @Time : 2022/5/16 9:57
# @Author : Scotty1373
# @File : utils.py
# @Software : PyCharm
import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from Envs.heat_map import normalize


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
    return img.copy()


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
