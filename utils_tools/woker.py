# -*- coding: utf-8 -*-
import numpy as np
from Envs.sea_env_without_orient import RoutePlan
from PPO.PPO import PPO, PPO_Buffer, SkipEnvFrame
from utils_tools.common import seed_torch
from utils_tools.utils import state_frame_overlay, pixel_based, img_proc, first_init
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch.multiprocessing as mp
import copy
import seaborn as sns
import torch

TIME_BOUNDARY = 500
IMG_SIZE = (80, 80)
IMG_SIZE_RENDEER = 480


def trace_trans(vect, *, ratio=IMG_SIZE_RENDEER/16):
    remap_vect = np.array((vect[0] * ratio + (IMG_SIZE_RENDEER / 2), (-vect[1] * ratio) + IMG_SIZE_RENDEER), dtype=np.uint16)
    return remap_vect


class worker(mp.Process):
    def __init__(self, args, name, worker_id, g_net_pi, g_net_v, pipe_line, tb_logger=None):
        super(worker, self).__init__()
        self.config = args
        self.workerID = worker_id
        self.name = f'{name}'
        self.g_net_pi = g_net_pi
        self.g_net_v = g_net_v
        self.pipe_line = pipe_line
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
        env = SkipEnvFrame(env, args.frame_skipping)
        assert isinstance(args.batch_size, int)
        seed_torch(seed=25532)
        device = torch.device('cpu')

        # 子线程显示当前环境heatmap
        fig, ax1 = plt.subplots(1, 1)
        sns.heatmap(env.env.heat_map, ax=ax1)
        fig.suptitle('reward shaping heatmap')
        # self.tb_logger.add_figure(f'worker{self.name}/heatmap', fig)

        agent = PPO(state_dim=args.frame_overlay * args.state_length,
                    action_dim=2,
                    batch_size=args.batch_size,
                    overlay=args.frame_overlay,
                    device=device)

        self.pull_from_global(agent)

        ep_history = []
        """agent探索轨迹追踪"""
        env.reset()
        trace_image = env.render(mode='rgb_array')
        trace_image = Image.fromarray(trace_image)
        trace_path = ImageDraw.Draw(trace_image)

        done = True

        # NameSpace
        trace_history = None
        pixel_obs = None
        obs = None

        for epoch in range(args.epochs):
            reward_history = 0
            entropy_acc_history = 0
            entropy_ori_history = 0
            buffer = PPO_Buffer()
            if done:
                """轨迹记录"""
                trace_history, pixel_obs, obs, done = first_init(env, args)

            for t in range(args.max_timestep):
                if done:
                    # 单幕结束显示轨迹
                    trace_path.line(trace_history, width=1, fill='black')
                    trace_history, pixel_obs, obs, done = first_init(env, args)
                act, logprob, dist = agent.get_action((pixel_obs, obs))
                # 环境交互
                pixel_obs_t1_ori, obs_t1, reward, done, _ = env.step(act.squeeze())
                pixel_obs_t1 = img_proc(pixel_obs_t1_ori)
                # 随机漫步如果为1则不进行数据庞拼接
                if args.frame_overlay == 1:
                    pass
                else:
                    obs_t1 = state_frame_overlay(obs_t1, obs, args.frame_overlay)
                    pixel_obs_t1 = pixel_based(pixel_obs_t1, pixel_obs, args.frame_overlay)

                if not args.pre_train:
                    # 状态存储
                    agent.state_store_memory(pixel_obs, obs, act, reward, logprob)
                    # 防止最后一次数据未被存储进buffer
                    if done or t == args.max_timestep - 1:
                        pixel_state, vect_state, action, logprob, d_reward, adv = agent.get_trjt(pixel_obs_t1, obs_t1, done)
                        buffer.collect_trajorbatch(pixel_state, vect_state, action, logprob, d_reward, adv)
                        agent.memory.clear()

                entropy_temp = dist.entropy().cpu().numpy().squeeze()
                entropy_acc = entropy_temp[0].item()
                entropy_ori = entropy_temp[1].item()
                # 记录timestep, reward＿sum
                agent.t += 1
                obs = obs_t1
                pixel_obs = pixel_obs_t1
                reward_history += reward
                entropy_acc_history += entropy_acc
                entropy_ori_history += entropy_ori

                trace_history.append(tuple(trace_trans(env.env.ship.position)))

            """管道发送buffer，并清空buffer"""
            self.pipe_line.send(buffer)
            # 从global取回参数
            try:
                self.pull_from_global(agent)
            except RuntimeError as e:
                print('pull parameter error')

            ep_history.append(reward_history)
            log_ep_text = {'epochs': epoch,
                           'time_step': agent.t,
                           'ep_reward': reward_history,
                           'entropy_acc_mean': entropy_acc_history / args.max_timestep,
                           'entropy_ori_mean': entropy_ori_history / args.max_timestep}

            # tensorboard logger
            # self.tb_logger.add_scalar(tag=f'Reward_{self.name}/ep_reward',
            #                           scalar_value=reward_history,
            #                           global_step=epoch)
            # self.tb_logger.add_scalar(tag=f'Reward_{self.name}/ep_entropy_acc',
            #                           scalar_value=log_ep_text["entropy_acc_mean"],
            #                           global_step=epoch)
            # self.tb_logger.add_scalar(tag=f'Reward_{self.name}/ep_entropy_ori',
            #                           scalar_value=log_ep_text["entropy_ori_mean"],
            #                           global_step=epoch)
            # self.tb_logger.add_image(tag=f'Image_{self.name}/Trace',
            #                          img_tensor=np.array(trace_image),
            #                          global_step=epoch,
            #                          dataformats='HWC')
        env.close()
        self.pipe_line.send(None)
        self.pipe_line.close()

    def pull_from_global(self, subprocess):
        self.hard_update(self.g_net_pi, subprocess.pi)
        self.hard_update(self.g_net_v, subprocess.v)

    @staticmethod
    def hard_update(model, target_model):
        weight_model = copy.deepcopy(model.state_dict())
        target_model.load_state_dict(weight_model)

