# -*- coding: utf-8 -*-
import argparse
import gc

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Envs.sea_env_without_orient import RoutePlan
from wrapper.wrapper import SkipEnvFrame
from TD3.TD3 import TD3
from utils_tools.common import TIMESTAMP, seed_torch
from utils_tools.utils import state_frame_overlay, pixel_based, first_init, trace_trans
from utils_tools.utils import ReplayBuffer

TIME_BOUNDARY = 500
IMG_SIZE = (80, 80)
IMG_SIZE_RENDEER = 480


def parse_args():
    parser = argparse.ArgumentParser(
        description='TD3 config option')
    parser.add_argument('--epochs',
                        help='Training epoch',
                        default=600,
                        type=int)
    parser.add_argument('--train',
                        help='Train or not',
                        default=True,
                        type=bool)
    parser.add_argument('--pre_train',
                        help='Pretrained?',
                        default=False,
                        type=bool)
    parser.add_argument('--checkpoint',
                        help='If pre_trained is True, this option is pretrained ckpt path',
                        default='./log/1659972659/save_model_ep800.pth',
                        type=str)
    parser.add_argument('--max_timestep',
                        help='Maximum time step in a single epoch',
                        default=512,
                        type=int)
    parser.add_argument('--seed',
                        help='environment initialization seed',
                        default=42,
                        type=int)
    parser.add_argument('--batch_size',
                        help='training batch size',
                        default=64,
                        type=int)
    parser.add_argument('--frame_skipping',
                        help='random walk frame skipping',
                        default=4,
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
                        default=2,
                        type=int)
    parser.add_argument('--pixel_state',
                        help='Image-Based Status',
                        default=False,
                        type=bool)
    parser.add_argument('--device',
                        help='data device',
                        default='cpu',
                        type=str)
    parser.add_argument('--replay_buffer_size',
                        help='Replay Buffer Size',
                        default=32000,
                        type=int)
    args = parser.parse_args()
    return args


def main(args):
    args = args
    seed_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # Iter log初始化
    # logger_iter = log2json(filename='train_log_iter', type_json=True)
    # # epoch log初始化
    # logger_ep = log2json(filename='train_log_ep', type_json=True)
    # tensorboard初始化
    tb_logger = SummaryWriter(log_dir=f"./log/{TIMESTAMP}_td3_random_start", flush_secs=120)
    replay_buffer = ReplayBuffer(max_lens=args.replay_buffer_size,
                                 frame_overlay=args.frame_overlay,
                                 state_length=args.state_length,
                                 action_dim=2,
                                 device=device)

    # 是否随机初始化种子
    if args.seed is not None:
        seed = args.seed
    else:
        seed = None

    # 环境与agent初始化
    env = RoutePlan(barrier_num=5, seed=seed, ship_pos_fixed=False)
    # env.seed(13)
    env = SkipEnvFrame(env, args.frame_skipping)
    assert isinstance(args.batch_size, int)

    """初始化agent"""
    agent = TD3(frame_overlay=args.frame_overlay,
                state_length=args.state_length,
                action_dim=2,
                batch_size=args.batch_size,
                overlay=args.frame_overlay,
                device=device,
                train=args.train,
                logger=tb_logger)

    def heatmap4tb_logger(*, ep=0):
        # 子线程显示当前环境heatmap
        fig, ax1 = plt.subplots(1, 1)
        sns.heatmap(env.env.heat_map.T, ax=ax1).invert_yaxis()
        fig.suptitle(f'reward shaping heatmap_{agent.stage}')
        tb_logger.add_figure(tag='figure',
                             figure=fig,
                             global_step=ep)

    # heatmap plot in tensorboard logger
    heatmap4tb_logger()

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
    # ep_history = []

    # NameSpace
    trace_history = None
    pixel_obs = None
    obs = None

    epochs = tqdm(range(args.epochs + 1), leave=False, position=0, colour='green')
    for epoch in epochs:
        reward_history = 0
        """***********这部分作为重置并没有起到训练连接的作用， 可删除if判断***********"""
        if done:
            """轨迹记录"""
            trace_history, pixel_obs, obs, done = first_init(env, args)

        # timestep 样本收集
        steps = tqdm(range(0, args.max_timestep), leave=False, position=1, colour='red')
        for t in steps:
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
                replay_buffer.add(pixel_obs, pixel_obs_t1, obs, obs_t1, reward, act, done)
                agent.update(replay_buffer)

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

            # 单幕数据收集完毕
            if done:
                # 单幕结束显示轨迹
                trace_path.line(trace_history, width=1, fill='black')
                trace_history, pixel_obs, obs, done = first_init(env, args)

        # ep_history.append(reward_history)
        # log_ep_text = {'epochs': epoch,
        #                'time_step': agent.t,
        #                'ep_reward': reward_history}
        agent.ep += 1
        agent.reset_noise()

        # tensorboard logger
        tb_logger.add_scalar(tag=f'Reward/ep_reward_{agent.stage}',
                             scalar_value=reward_history,
                             global_step=epoch)
        tb_logger.add_image(tag=f'Image/Trace_{agent.stage}',
                            img_tensor=np.array(trace_image),
                            global_step=epoch,
                            dataformats='HWC')
        # 环境重置
        if not epoch % 25:
            env.close()
            env = RoutePlan(barrier_num=5, seed=seed, ship_pos_fixed=False)
            env = SkipEnvFrame(env, args.frame_skipping)
            if agent.logger_reload:
                # reload changed heatmap
                heatmap4tb_logger(ep=epoch)
                # reload route trace image
                trace_image = env.render(mode='rgb_array')
                trace_image = Image.fromarray(trace_image)
                trace_path = ImageDraw.Draw(trace_image)
            agent.save_model(f'./log/{TIMESTAMP}_td3_random_start/save_model_ep{epoch}_opt-{args.seed}_{args.batch_size}_{args.state_length}.pth')
    env.close()
    tb_logger.close()


if __name__ == '__main__':
    config = parse_args()
    main(config)
