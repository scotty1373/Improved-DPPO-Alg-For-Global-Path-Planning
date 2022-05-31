# -*- coding: utf-8 -*-
import numpy as np
from PPO.PPO import PPO, PPO_Buffer
from utils_tools.common import log2json, TIMESTAMP, seed_torch
from utils_tools import worker
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
                        default=2000)
    parser.add_argument('--pre_train',
                        help='Pretrained?',
                        default=False)
    parser.add_argument('--checkpoint',
                        help='If pre_traine is True, this option is pretrained ckpt path',
                        default=None)
    parser.add_argument('--max_timestep',
                        help='Maximum time step in a single epoch',
                        default=256)
    parser.add_argument('--seed',
                        help='environment initialization seed',
                        default=42)
    parser.add_argument('--batch_size',
                        help='training batch size',
                        default=64)
    parser.add_argument('--frame_skipping',
                        help='random walk frame skipping',
                        default=3)
    parser.add_argument('--frame_overlay',
                        help='data frame overlay',
                        default=3)
    # parser.add_argument('--state_length',
    #                     help='state data vector length',
    #                     default=5+24*2)
    parser.add_argument('--state_length',
                        help='state data vector length',
                        default=3)
    parser.add_argument('--pixel_state',
                        help='Image-Based Status',
                        default=False)
    parser.add_argument('--device',
                        help='data device',
                        default='cpu')
    parser.add_argument('--worker_num',
                        help='worker number',
                        default=3)
    args = parser.parse_args()
    return args


def main(args):
    args = args
    seed_torch(seed=25532)
    device = torch.device('cuda')
    torch.multiprocessing.set_start_method('spawn')

    # # Iter log初始化
    # logger_iter = log2json(filename='train_log_iter', type_json=True)
    # # epoch log初始化
    # logger_ep = log2json(filename='train_log_ep', type_json=True)
    # tensorboard初始化
    tb_logger = SummaryWriter(log_dir=f"./log/{TIMESTAMP}", flush_secs=120)

    global_ppo = PPO(state_dim=args.frame_overlay * args.state_length,
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

    pipe_r, pipe_w = zip(*[mp.Pipe() for _ in range(args.worker_num)])
    worker_list = [worker(args, f'worker{i}', i, global_ppo.pi, global_ppo.v, pipe_w[i], TIMESTAMP) for i in range(args.worker_num)]
    [worker_idx.start() for worker_idx in worker_list]

    epochs = tqdm(range(args.epochs), leave=False, position=0, colour='green')
    for epoch in epochs:
        steps = tqdm(range(0, args.worker_num), leave=False, position=1, colour='red')
        # 从子线程中获取数据
        for step_i in steps:
            subprocess_buffer = pipe_r[step_i].recv()
            if subprocess_buffer is None:
                worker_list[step_i].join()
            else:
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

    global_ppo.save_model(f'./log/{TIMESTAMP}/save_model_ep{epoch}.pth')
    tb_logger.close()


if __name__ == '__main__':
    config = parse_args()
    main(config)
