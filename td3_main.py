# -*- coding: utf-8 -*-
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Envs.sea_env_without_orient import RoutePlan
from PPO.wrapper import SkipEnvFrame
from TD3.TD3 import TD3
from utils_tools.common import TIMESTAMP, seed_torch
from utils_tools.utils import state_frame_overlay, pixel_based, first_init, trace_trans

TIME_BOUNDARY = 500
IMG_SIZE = (80, 80)
IMG_SIZE_RENDEER = 480


def parse_args():
    parser = argparse.ArgumentParser(
        description='PPO config option')
    parser.add_argument('--epochs',
                        help='Training epoch',
                        default=2000)
    parser.add_argument('--train',
                        help='Train or not',
                        default=True,
                        type=bool)
    parser.add_argument('--pre_train',
                        help='Pretrained?',
                        default=True,
                        type=bool)
    parser.add_argument('--checkpoint',
                        help='If pre_trained is True, this option is pretrained ckpt path',
                        default='./log/1657730384/save_model_ep344.pth')
    parser.add_argument('--max_timestep',
                        help='Maximum time step in a single epoch',
                        default=512)
    parser.add_argument('--seed',
                        help='environment initialization seed',
                        default=42)
    parser.add_argument('--batch_size',
                        help='training batch size',
                        default=128)
    parser.add_argument('--frame_skipping',
                        help='random walk frame skipping',
                        default=4)
    parser.add_argument('--frame_overlay',
                        help='data frame overlay',
                        default=4)
    # parser.add_argument('--state_length',
    #                     help='state data vector length',
    #                     default=5+24*2)
    parser.add_argument('--state_length',
                        help='state data vector length',
                        default=2)
    parser.add_argument('--pixel_state',
                        help='Image-Based Status',
                        default=False,
                        type=bool)
    parser.add_argument('--device',
                        help='data device',
                        default='cpu')
    parser.add_argument('--worker_num',
                        help='worker number',
                        default=5)
    args = parser.parse_args()
    return args


def main(args):
    args = args
    seed_torch()
    device = torch.device('cuda')

    # # Iter log初始化
    # logger_iter = log2json(filename='train_log_iter', type_json=True)
    # # epoch log初始化
    # logger_ep = log2json(filename='train_log_ep', type_json=True)
    # tensorboard初始化
    tb_logger = SummaryWriter(log_dir=f"./log/{TIMESTAMP}", flush_secs=120)

    # 是否随机初始化种子
    if args.seed is not None:
        seed = args.seed
    else:
        seed = None

    # 环境与agent初始化
    env = RoutePlan(barrier_num=3, seed=seed, ship_pos_fixed=True)
    # env.seed(13)
    env = SkipEnvFrame(env, args.frame_skipping)
    assert isinstance(args.batch_size, int)

    # 子线程显示当前环境heatmap
    fig, ax1 = plt.subplots(1, 1)
    sns.heatmap(env.env.heat_map, ax=ax1)
    fig.suptitle('reward shaping heatmap')
    tb_logger.add_figure('figure', fig)

    """初始化agent"""
    agent = TD3(frame_overlay=args.frame_overlay,
                state_length=args.state_length,
                action_dim=2,
                batch_size=args.batch_size,
                overlay=args.frame_overlay,
                device=device,
                train=args.train,
                logger=tb_logger)

    # pretrained 选项，载入预训练模型
    if args.pre_train:
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
            agent.load_model(checkpoint)

    """初始化agent探索轨迹追踪"""
    env.reset()
    trace_image = env.render(mode='rgb_array')
    trace_image = Image.fromarray(trace_image)
    trace_path = ImageDraw.Draw(trace_image)

    done = True
    ep_history = []

    # NameSpace
    trace_history = None
    pixel_obs = None
    obs = None

    epochs = tqdm(range(args.epochs), leave=False, position=0, colour='green')
    for epoch in epochs:
        reward_history = 0
        """***********这部分作为重置并没有起到训练连接的作用， 可删除if判断***********"""
        if done:
            """轨迹记录"""
            trace_history, pixel_obs, obs, done = first_init(env, args)

        # timestep 样本收集
        steps = tqdm(range(0, args.max_timestep), leave=False, position=1, colour='red')
        for t in steps:
            # 单幕数据收集完毕
            if done:
                # 单幕结束显示轨迹
                trace_path.line(trace_history, width=1, fill='black')
                trace_history, pixel_obs, obs, done = first_init(env, args)
            act = agent.get_action(pixel_obs, obs)
            # 环境交互
            pixel_obs_t1, obs_t1, reward, done, _ = env.step(act.squeeze())

            # 随机漫步如果为1则不进行数据庞拼接
            if args.frame_overlay == 1:
                pass
            else:
                obs_t1 = state_frame_overlay(obs_t1, obs, args.frame_overlay)
                pixel_obs_t1 = pixel_based(pixel_obs_t1, pixel_obs, args.frame_overlay)

            if args.train:
                # 达到最大timestep则认为单幕完成
                if t == args.max_timestep - 1:
                    done = True
                # 状态存储
                agent.state_store_memory(pixel_obs, obs, act, reward, pixel_obs_t1, obs_t1, done)
                agent.update()

            # 记录timestep, reward＿sum
            agent.t += 1
            obs = obs_t1
            pixel_obs = pixel_obs_t1
            reward_history += reward

            trace_history.append(tuple(trace_trans(env.env.ship.position)))
            steps.set_description(f"epochs: {epoch}, "
                                  f"time_step: {agent.t}, "
                                  f"ep_reward: {reward_history}, "
                                  f"acc: {act[..., 0].item():.2f}, "
                                  f"ori: {act[..., 1].item():.2f}, "
                                  f"reward: {reward:.2f}, "
                                  f"done: {done}, "
                                  f"actor_loss: {agent.actor_loss_history:.2f}, "
                                  f"critic_loss: {agent.critic_loss_history:.2f}")

        ep_history.append(reward_history)
        log_ep_text = {'epochs': epoch,
                       'time_step': agent.t,
                       'ep_reward': reward_history}
        agent.ep += 1

        # tensorboard logger
        tb_logger.add_scalar(tag=f'Reward/ep_reward',
                             scalar_value=reward_history,
                             global_step=epoch)
        tb_logger.add_image(tag=f'Image/Trace',
                            img_tensor=np.array(trace_image),
                            global_step=epoch,
                            dataformats='HWC')
        # 环境重置
        if not epoch % 50:
            env.close()
            env = RoutePlan(barrier_num=3, seed=seed, ship_pos_fixed=True)
            env = SkipEnvFrame(env, args.frame_skipping)
    env.close()
    tb_logger.close()
    agent.save_model(f'./log/{TIMESTAMP}/save_model_ep{epoch}.pth')


if __name__ == '__main__':
    config = parse_args()
    main(config)
