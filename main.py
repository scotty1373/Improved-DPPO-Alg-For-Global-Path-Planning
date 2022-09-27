# -*- coding: utf-8 -*-
import sys
import time

import numpy as np
from PPO.PPO import PPO, PPO_Buffer
from utils_tools.common import log2json, TIMESTAMP, seed_torch
from utils_tools.worker import worker
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from tqdm import tqdm
import torch
import argparse

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
                        default=True,
                        type=bool)
    parser.add_argument('--pre_train',
                        help='Pretrained?',
                        default=False,
                        type=bool)
    parser.add_argument('--checkpoint',
                        help='If pre_trained is True, this option is pretrained ckpt path',
                        default="./log/1664198401/save_model_ep550.pth",
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
                        default=256,
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
                        default=5,
                        type=int)
    args = parser.parse_args()
    return args


def main(args):
    args = args
    seed_torch()
    device = torch.device('cuda')
    torch.multiprocessing.set_start_method('spawn')

    # # Iter log初始化
    # logger_iter = log2json(filename='train_log_iter', type_json=True)
    # # epoch log初始化
    # logger_ep = log2json(filename='train_log_ep', type_json=True)
    # tensorboard初始化
    tb_logger = SummaryWriter(log_dir=f"./log/{TIMESTAMP}", flush_secs=120)

    global_ppo = PPO(frame_overlay=args.frame_overlay,
                     state_length=args.state_length,
                     action_dim=2,
                     batch_size=args.batch_size,
                     overlay=args.frame_overlay,
                     device=device,
                     logger=tb_logger)

    # 是否从预训练结果中载入ckpt
    if args.pre_train:
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
            global_ppo.load_model(checkpoint)

    training_buffer = PPO_Buffer()

    # 进程共享Event
    event = mp.Event()

    pipe_r, pipe_w = zip(*[mp.Pipe(duplex=False) for _ in range(args.worker_num)])
    worker_list = [worker(args, f'worker{i}', i, global_ppo.pi, global_ppo.v, pipe_w[i], event, TIMESTAMP) for i in range(args.worker_num)]
    [worker_idx.start() for worker_idx in worker_list]

    # Event Reset
    event.set()
    event.clear()

    epochs = tqdm(range(args.epochs), leave=False, position=0, colour='green')
    for epoch in epochs:
        steps = tqdm(range(0, args.worker_num), leave=False, position=1, colour='red')
        # 从子线程中获取数据
        for step_i in steps:
            subprocess_buffer = pipe_r[step_i].recv()
            training_buffer.collect_batch(subprocess_buffer.pixel_state,
                                          subprocess_buffer.vect_state,
                                          subprocess_buffer.action,
                                          subprocess_buffer.logprob,
                                          subprocess_buffer.d_reward,
                                          subprocess_buffer.adv)
            del subprocess_buffer

        # 参数更新
        global_ppo.update(training_buffer, args)
        # lr_Scheduler
        global_ppo.a_sch.step()
        global_ppo.c_sch.step()
        global_ppo.ep += 1

        if epoch % 50 == 0:
            global_ppo.save_model(f'./log/{TIMESTAMP}/save_model_ep{epoch}.pth')

    tb_logger.close()
    # subProcess enable
    event.set()
    # join all subProcess
    [i.join() for i in worker_list]
    [i.terminate() for i in worker_list]


if __name__ == '__main__':
    config = parse_args()
    main(config)
