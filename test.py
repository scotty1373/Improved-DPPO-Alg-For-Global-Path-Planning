# -*- coding: utf-8 -*-
import os.path
import sys

import numpy as np
from Envs.sea_env_without_orient import RoutePlan
from PPO.PPO import PPO, SkipEnvFrame
from utils_tools.utils import state_frame_overlay, pixel_based, img_proc, first_init
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from utils_tools.common import TIMESTAMP
import seaborn as sns
from utils_tools.common import seed_torch
import argparse
import torch

TIME_BOUNDARY = 500
IMG_SIZE = (80, 80)
IMG_SIZE_RENDEER = 480


def parse_args():
    parser = argparse.ArgumentParser(
        description='PPO config option')
    parser.add_argument('--epochs',
                        help='Training epoch',
                        default=600,
                        type=int)
    parser.add_argument('--train',
                        help='Train or not',
                        default=False,
                        type=bool)
    parser.add_argument('--pre_train',
                        help='Pretrained?',
                        default=True,
                        type=bool)
    parser.add_argument('--checkpoint',
                        help='If pre_trained is True, this option is pretrained ckpt path',
                        default="./log/1662016523/save_model_ep550.pth",
                        type=str)
    parser.add_argument('--max_timestep',
                        help='Maximum time step in a single epoch',
                        default=256,
                        type=int)
    parser.add_argument('--seed',
                        help='environment initialization seed',
                        default=42,
                        type=int)
    parser.add_argument('--batch_size',
                        help='training batch size',
                        default=128,
                        type=int)
    parser.add_argument('--frame_skipping',
                        help='random walk frame skipping',
                        default=3,
                        type=int)
    parser.add_argument('--frame_overlay',
                        help='data frame overlay',
                        default=4,
                        type=int)
    # parser.add_argument('--state_length',
    #                     help='state data vector length',
    #                     default=5+24*2)
    parser.add_argument('--state_length',
                        help='state data vector length',
                        default=4,
                        type=int)
    parser.add_argument('--pixel_state',
                        help='Image-Based Status',
                        default=False,
                        type=int)
    parser.add_argument('--device',
                        help='data device',
                        default='cpu',
                        type=str)
    parser.add_argument('--worker_num',
                        help='worker number',
                        default=1,
                        type=int)
    args = parser.parse_args()
    return args


def trace_trans(vect, *, ratio=IMG_SIZE_RENDEER/16):
    remap_vect = np.array((vect[0] * ratio + (IMG_SIZE_RENDEER / 2), (-vect[1] * ratio) + IMG_SIZE_RENDEER), dtype=np.uint16)
    return remap_vect


def main(args):
    args = args
    seed_torch()
    device = torch.device('cuda')
    seed = args.seed

    if not os.path.exists(f'f./log/{TIMESTAMP}'):
        os.makedirs(f'./log/{TIMESTAMP}/')

    # 环境与agent初始化
    env = RoutePlan(barrier_num=5, seed=seed, ship_pos_fixed=True, worker_id=None, test=True)
    # env.seed(13)
    env = SkipEnvFrame(env, args.frame_skipping)
    assert isinstance(args.batch_size, int)
    # seed_torch(seed=25532)
    device = torch.device('cpu')

    # 子线程显示当前环境heatmap
    # fig, ax1 = plt.subplots(1, 1)
    # sns.heatmap(env.env.heat_map.T, ax=ax1).invert_yaxis()
    # fig.suptitle('reward shaping heatmap')

    agent = PPO(frame_overlay=args.frame_overlay,
                state_length=args.state_length,
                action_dim=2,
                batch_size=args.batch_size,
                overlay=args.frame_overlay,
                device=device,
                logger=None)

    # 是否从预训练结果中载入ckpt
    if args.pre_train:
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
            agent.load_model(checkpoint)

    ep_history = []
    done = True

    # NameSpace
    trace_history = None
    pixel_obs = None
    obs = None

    for epoch in range(args.epochs):
        reward_history = 0
        entropy_acc_history = 0
        entropy_ori_history = 0
        """***********这部分作为重置并没有起到训练连接的作用， 可删除if判断***********"""
        if done:
            """轨迹记录"""
            trace_history, pixel_obs, obs, done = first_init(env, args)

        for t in range(args.max_timestep):
            if done:
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

            if done or t == args.max_timestep - 1:
                done = True

            entropy_temp = dist.mean.cpu().numpy().squeeze()
            entropy_acc = entropy_temp[0].item()
            entropy_ori = entropy_temp[1].item()
            # 记录timestep, reward＿sum
            agent.t += 1
            obs = obs_t1
            pixel_obs = pixel_obs_t1
            reward_history += reward
            entropy_acc_history += entropy_acc
            entropy_ori_history += entropy_ori

            # tb_logger.add_scalar(tag='test/entropy_acc',
            #                      scalar_value=entropy_acc,
            #                      global_step=epoch * args.max_timestep + t)
            # tb_logger.add_scalar(tag='test/entropy_ori',
            #                      scalar_value=entropy_ori,
            #                      global_step=epoch * args.max_timestep + t)

            trace_history.append(tuple(trace_trans(env.env.ship.position)))
            if done:
                trace_image = env.render(mode='rgb_array')
                trace_image = Image.fromarray(trace_image)
                trace_path = ImageDraw.Draw(trace_image)
                trace_path.point(trace_history, fill='Black')
                # trace_path.line(trace_history, width=10, fill='blue')
                trace_image.save(f'./log/{TIMESTAMP}/track_{t}.png', quality=95)
            if env.env.end:
                sys.exit()

        ep_history.append(reward_history)
        log_ep_text = {'epochs': epoch,
                       'time_step': agent.t,
                       'ep_reward': reward_history,
                       'entropy_acc_mean': entropy_acc_history / args.max_timestep,
                       'entropy_ori_mean': entropy_ori_history / args.max_timestep}

    env.close()


if __name__ == '__main__':
    main(parse_args())