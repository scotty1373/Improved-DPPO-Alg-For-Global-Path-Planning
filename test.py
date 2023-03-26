# -*- coding: utf-8 -*-
# @Time    : 2023/3/25 19:01
# @Author  : Scotty
# @FileName: test.py
# @Software: PyCharm
import argparse
import os
import time
import cv2

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from Envs.sea_env_without_orient import RoutePlan, SHIP_POSITION, Distance_Cacul
from TD3.TD3 import TD3
from utils_tools.common import TIMESTAMP, seed_torch
from utils_tools.utils import state_frame_overlay, pixel_based, first_init, trace_trans
from wrapper.wrapper import SkipEnvFrame

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
                        default=False,
                        type=bool)
    parser.add_argument('--pre_train',
                        help='Pretrained?',
                        default=True,
                        type=bool)
    parser.add_argument('--checkpoint',
                        help='If pre_trained is True, this option is pretrained ckpt path',
                        default='./log/1679746163/save_model_ep200_opt-42_64_4.pth',
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
                        default=4,
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
    save_forever = False
    device = torch.device('cpu')

    # # Iter log初始化
    # logger_iter = log2json(filename='train_log_iter', type_json=True)
    # # epoch log初始化
    # logger_ep = log2json(filename='train_log_ep', type_json=True)

    # 是否随机初始化种子
    if args.seed is not None:
        seed = args.seed
    else:
        seed = None

    if not os.path.exists(f'./log/{TIMESTAMP}_test'):
        os.makedirs(f'./log/{TIMESTAMP}_test')

    # 环境与agent初始化
    env = RoutePlan(barrier_num=5, seed=seed, ship_pos_fixed=True, test=True)
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
                logger=None)

    # 载入预训练模型
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
    time_cost = 0.0

    for epoch in range(1):
        reward_history = 0
        dist_history = {0: [],
                        1: [],
                        2: [],
                        3: [],
                        4: [],
                        5: []}

        # timestep 样本收集
        for t in range(5120):
            if done:
                trace_history, pixel_obs, obs, done = first_init(env, args)
                trace_image = env.render(mode='rgb_array')
                time_cost = 0.0
                opts_counter = 0

            # **************************************************************
            start_time = time.time()
            # **************************************************************
            act = agent.get_action(pixel_obs, obs)
            # **************************************************************
            time_cost += time.time() - start_time
            # **************************************************************
            # 环境交互
            pixel_obs_t1, obs_t1, reward, done, _ = env.step(act.squeeze())
            opts_counter += 1
            # 随机漫步如果为1则不进行数据庞拼接
            if args.frame_overlay == 1:
                pass
            else:
                obs_t1 = state_frame_overlay(obs_t1, obs, args.frame_overlay)
                pixel_obs_t1 = pixel_based(pixel_obs_t1, pixel_obs, args.frame_overlay)

            if done or t == args.max_timestep - 1:
                done = True

            agent.t += 1
            obs = obs_t1
            pixel_obs = pixel_obs_t1

            # 追踪点记录
            trace_history.append(tuple(trace_trans(env.env.ship.position)))

            # if env is done
            if done:
                # get current position index
                index = SHIP_POSITION.index(env.env.iter_ship_pos.val)
                # env terminated by complete progress
                if env.env.game_over:
                    dist = get_dist(trace_history)
                    dist_history[index].append(dist)
                    if save_forever or dist <= min(dist_history[index]):
                        # added reach point center position
                        trace_history.append(tuple(trace_trans(env.env.reach_area.position)))

                        trace_image = Image.fromarray(trace_image)
                        trace_path = ImageDraw.Draw(trace_image)
                        trace_path.point(trace_history, fill='Black')
                        trace_path.line(trace_history, width=1, fill='blue')
                        # cv2_lines(np.array(trace_image), trace_history)
                        trace_image.save(f'./log/{TIMESTAMP}_test/track_{index}_{t}.png', quality=95)
                        print(f"access point: {index}, path length: {get_dist(trace_history)},"
                              f"time cost: {time_cost}, opts: {opts_counter}")
                # env terminated by false
                else:
                    dist_history[index].append(1000.0)

        min_length = 0
        # calculate the successful rate
        for i in dist_history:
            print(f'env {i}, RoutePlan successful rate: {success_plan_rate(dist_history[i], 1000.)}')
            dist_history[i] = np.array(dist_history[i])
            # 五个区域中最少成功次数，为做pandas dataframe长度匹配
            min_length = max(dist_history[i].shape[0], min_length)

        for idx in dist_history:
            dist_history[idx] = np.random.choice(dist_history[idx], min_length)

        df = pd.DataFrame.from_dict(dist_history)
        df.to_csv(f"./log/{TIMESTAMP}_test/benchmark.csv")
        break
    env.close()


def success_plan_rate(seq, error_bound):
    seq = np.array(seq)
    error_seq = np.where(seq == error_bound)[0]
    return 1 - error_seq.shape[0]/seq.shape[0]


def cv2_lines(img: np.ndarray, trace_path: list):
    # line
    # for idx, pos in enumerate(trace_path[1:]):
    #     img = cv2.line(img, pos, trace_path[idx], (255, 0, 0), 1)
    # mini circle
    for pos in trace_path:
        img = cv2.circle(img, pos, 1, (0, 0, 255), 0)
    tmp = Image.fromarray(img)
    tmp.show()


def get_dist(trace_history):
    dist_record = 0
    for idx, pos in enumerate(trace_history[1:]):
        dist = Distance_Cacul(trace_history[idx], pos)
        dist_record += dist
    return dist_record


if __name__ == '__main__':
    config = parse_args()
    main(config)
