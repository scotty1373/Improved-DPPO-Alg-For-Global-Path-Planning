# -*- coding: utf-8 -*-
import numpy as np

from Envs.sea_env_without_orient import RoutePlan
from PPO.PPO import PPO
from utils_tools.common import log2json, dirs_creat, TIMESTAMP, seed_torch
from torch.utils.tensorboard import SummaryWriter
from utils_tools.utils import state_frame_overlay, pixel_based, img_proc
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch.multiprocessing as mp

import seaborn as sns
from tqdm import tqdm
import torch

TIME_BOUNDARY = 500
IMG_SIZE = (80, 80)
IMG_SIZE_RENDEER = 480


def trace_trans(vect, *, ratio=IMG_SIZE_RENDEER/16):
    remap_vect = np.array((vect[0] * ratio + (IMG_SIZE_RENDEER / 2), (-vect[1] * ratio) + IMG_SIZE_RENDEER), dtype=np.uint16)
    return remap_vect

class worker(mp.Process):
    def __init__(self, args, name, g_net, g_opt, global_ep, global_r, global_res, worker_id, tb_logger):
        super(worker, self).__init__()
        self.config = args
        self.workerID = worker_id
        self.name = f'{name}'
        self.g_net = g_net
        self.g_opt = g_opt
        self.global_ep, self.global_r, self.global_res = global_ep, global_r, global_res
        self.tb_logger = tb_logger
        

    def run(self):
        args = self.config
        # 是否随机初始化种子
        if args.seed is not None:
            seed = args.seed
        else:
            seed = None

        # 环境与agent初始化
        env = RoutePlan(barrier_num=3, seed=seed)
        env.seed(13)
        env.unwrapped
        assert isinstance(args.batch_size, int)
        # agent = PPO(state_dim=3*(7+24), action_dim=2, batch_size=args.batch_size)
        # seed_torch(seed=25535)
        device = torch.device('cuda')
        agent = PPO(state_dim=args.frame_overlay * args.state_length,
                    action_dim=2,
                    batch_size=args.batch_size,
                    overlay=args.frame_overlay,
                    device=device)

        ep_history = []
        """agent探索轨迹追踪"""
        env.reset()
        trace_image = env.render(mode='rgb_array')
        trace_image = Image.fromarray(trace_image)
        trace_path = ImageDraw.Draw(trace_image)

        for epoch in range(args.epochs):
            reward_history = 0
            entropy_acc_history = 0
            entropy_ori_history = 0
            """轨迹记录"""
            trace_history = []
            obs, _, done, _ = env.reset()
            '''利用广播机制初始化state帧叠加结构，不使用stack重复对数组进行操作'''
            obs = (np.ones((args.frame_overlay, args.state_length)) * obs).reshape(1, -1)
            pixel_obs_ori = env.render(mode='rgb_array')
            pixel_obs = img_proc(pixel_obs_ori) * np.ones((1, 3, 80, 80))

            for t in range(1, args.max_timestep * args.frame_skipping):
                # 是否进行可视化渲染
                env.render()
                if t % args.frame_skipping == 0:
                    act, logprob, dist = agent.get_action((pixel_obs, obs))
                    # 环境交互
                    obs_t1, reward, done, _ = env.step(act.squeeze(), t)
                    pixel_obs_t1_ori = env.render()
                    pixel_obs_t1 = img_proc(pixel_obs_t1_ori)
                    # 随机漫步如果为1则不进行数据庞拼接
                    if args.frame_overlay == 1:
                        pass
                    else:
                        obs_t1 = state_frame_overlay(obs_t1, obs, args.frame_overlay)
                        pixel_obs_t1 = pixel_based(pixel_obs_t1, pixel_obs, args.frame_overlay)
                    # 达到maxstep次数之后给予惩罚
                    if (t + args.frame_skipping) % args.max_timestep == 0:
                        done = True
                        reward = -10

                    if not args.pre_train:

                        # 状态存储
                        agent.state_store_memory(pixel_obs, obs, act, reward, logprob)

                        if t % (args.frame_skipping * agent.batch_size) == 0 or (
                                done and t % (agent.batch_size * args.frame_skipping) > 5):
                            pixel_state, vect_state, action, reward_nstep, logprob_nstep = zip(*agent.memory)

                            pixel_state = np.concatenate(pixel_state, axis=0)
                            vect_state = np.concatenate(vect_state, axis=0)
                            action = np.concatenate(action, axis=0)
                            reward_nstep = np.stack(reward_nstep, axis=0)
                            logprob_nstep = np.concatenate(logprob_nstep, axis=0)
                            # 动作价值计算
                            discount_reward = agent.decayed_reward((pixel_obs_t1, obs_t1), reward_nstep)

                            # 计算gae advantage
                            with torch.no_grad():
                                last_frame_pixel = torch.Tensor(pixel_obs_t1).to(device)
                                last_frame_vect = torch.Tensor(obs_t1).to(device)
                                last_val = agent.v(last_frame_pixel, last_frame_vect)
                            # 策略网络价值网络更新
                            agent.update(pixel_state, vect_state, action, logprob_nstep, discount_reward, reward_nstep,
                                         last_val, done)
                            # 清空存储池
                            agent.memory.clear()

                    entropy_temp = dist.entropy().cpu().numpy().squeeze()
                    # 记录timestep, reward＿sum
                    agent.t += 1
                    obs = obs_t1
                    pixel_obs = pixel_obs_t1
                    reward_history += reward
                    entropy_acc_history += entropy_temp[0].item()
                    entropy_ori_history += entropy_temp[1].item()
                    if done:
                        break

                else:
                    env.step(np.zeros(2, ), t)

                trace_history.append(tuple(trace_trans(env.ship.position)))

                if (t + 1) % (args.max_timestep * args.frame_skipping) == 0:
                    break

            # 单幕结束显示轨迹
            trace_path.line(trace_history, width=1, fill='black')

            # lr_Scheduler
            agent.a_sch.step()
            agent.c_sch.step()

            ep_history.append(reward_history)
            log_ep_text = {'epochs': epoch,
                           'time_step': agent.t,
                           'ep_reward': reward_history,
                           'entropy_acc_mean': entropy_acc_history / (t + 1),
                           'entropy_ori_mean': entropy_ori_history / (t + 1)}

            # tensorboard logger
            self.tb_logger.add_scalar(tag=f'Loss_{self.name}/ep_reward',
                                      scalar_value=reward_history,
                                      global_step=epoch)
            self.tb_logger.add_scalar(tag=f'Loss_{self.name}/ep_entropy_acc',
                                      scalar_value=log_ep_text["entropy_acc_mean"],
                                      global_step=epoch)
            self.tb_logger.add_scalar(tag=f'Loss_{self.name}/ep_entropy_ori',
                                      scalar_value=log_ep_text["entropy_ori_mean"],
                                      global_step=epoch)
            self.tb_logger.add_image(tag=f'Image_{self.name}/Trace',
                                     img_tensor=np.array(trace_image),
                                     global_step=epoch,
                                     dataformats='HWC')
        env.close()
