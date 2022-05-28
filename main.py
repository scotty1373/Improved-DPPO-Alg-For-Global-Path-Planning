# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from Envs.sea_env_without_orient import RoutePlan
from PPO.PPO import PPO, PPO_Buffer, SkipEnvFrame
from utils_tools.common import log2json, dirs_creat, TIMESTAMP, seed_torch
from torch.utils.tensorboard import SummaryWriter
from utils_tools.utils import state_frame_overlay, pixel_based, img_proc, first_init
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import seaborn as sns
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
                        default=512)
    parser.add_argument('--seed',
                        help='environment initialization seed',
                        default=None)
    parser.add_argument('--batch_size',
                        help='training batch size',
                        default=16)
    parser.add_argument('--frame_skipping',
                        help='random walk frame skipping',
                        default=4)
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
    args = parser.parse_args()
    return args


def trace_trans(vect, *, ratio=IMG_SIZE_RENDEER/16):
    remap_vect = np.array((vect[0] * ratio + (IMG_SIZE_RENDEER / 2), (-vect[1] * ratio) + IMG_SIZE_RENDEER), dtype=np.uint16)
    return remap_vect


def main(args):
    args = args
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
    # agent = PPO(state_dim=3*(7+24), action_dim=2, batch_size=args.batch_size)
    seed_torch(seed=25532)
    device = torch.device('cuda')

    # Iter log初始化
    logger_iter = log2json(filename='train_log_iter', type_json=True)
    # epoch log初始化
    logger_ep = log2json(filename='train_log_ep', type_json=True)
    # tensorboard初始化
    tb_logger = SummaryWriter(log_dir=f"./log/{TIMESTAMP}", flush_secs=120)
    fig, ax1 = plt.subplots(1, 1)
    sns.heatmap(env.env.heat_map, ax=ax1)
    fig.suptitle('reward shaping heatmap')
    tb_logger.add_figure('figure', fig)

    agent = PPO(state_dim=args.frame_overlay * args.state_length,
                action_dim=2,
                batch_size=args.batch_size,
                overlay=args.frame_overlay,
                device=device,
                logger=tb_logger)

    # 是否从预训练结果中载入ckpt
    if args.pre_train:
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
            agent.load_model(checkpoint)

    ep_history = []
    """agent探索轨迹追踪"""
    env.reset()
    trace_image = env.env.render(mode='rgb_array')
    trace_image = Image.fromarray(trace_image)
    trace_path = ImageDraw.Draw(trace_image)

    done = True
    epochs = tqdm(range(args.epochs), leave=False, position=0, colour='green')
    for epoch in epochs:
        reward_history = 0
        entropy_acc_history = 0
        entropy_ori_history = 0
        buffer = PPO_Buffer()
        if done:
            """轨迹记录"""
            trace_history, pixel_obs, obs, done = first_init(env, args)

        step = tqdm(range(1, args.max_timestep + 1), leave=False, position=1, colour='red')
        for t in step:
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

            # # 达到maxstep次数之后给予惩罚
            # if t % args.max_timestep == 0:
            #     done = True
            #     reward = -10

            if not args.pre_train:
                # 状态存储
                agent.state_store_memory(pixel_obs, obs, act, reward, logprob)

                if done or t == args.max_timestep:
                    pixel_state, vect_state, action, logprob, d_reward, adv = agent.get_trjt(pixel_obs_t1, obs_t1, done)
                    buffer.collect_traj(pixel_state, vect_state, action, logprob, d_reward, adv)
                    agent.memory.clear()

            entropy_temp = dist.entropy().cpu().numpy().squeeze()
            entropy_acc = entropy_temp[0].item()
            entropy_ori = entropy_temp[1].item()
            log_text = {'epochs': epoch,
                        'time_step': agent.t,
                        'reward': reward,
                        'entropy_acc': entropy_acc,
                        'entropy_ori': entropy_ori,
                        'acc': act.squeeze()[0].item(),
                        'ori': act.squeeze()[1].item(),
                        'actor_loss': agent.history_actor,
                        'critic_loss': agent.history_critic}
            step.set_description(f'epochs:{epoch}, '
                                 f'time_step:{agent.t}, '
                                 f'reward:{reward:.1f}, '
                                 f'et_acc: {log_text["entropy_acc"]:.1f}, '
                                 f'et_ori: {log_text["entropy_ori"]:.1f}, '
                                 f'acc:{log_text["acc"]:.1f}, '
                                 f'ori:{log_text["ori"]:.1f}, '
                                 f'lr:{agent.a_opt.state_dict()["param_groups"][0]["lr"]:.5f}, '
                                 f'ang_vel:{env.env.ship.angularVelocity:.1f}, '
                                 f'actor_loss:{agent.history_actor:.1f}, '
                                 f'critic_loss:{agent.history_critic:.1f}')
            # iter数据写入log文件
            logger_iter.write2json(log_text)

            # 记录timestep, reward＿sum
            agent.t += 1
            obs = obs_t1
            pixel_obs = pixel_obs_t1
            reward_history += reward
            entropy_acc_history += entropy_acc
            entropy_ori_history += entropy_ori

            trace_history.append(tuple(trace_trans(env.env.ship.position)))
        # 参数更新
        agent.update(buffer)

        # lr_Scheduler
        agent.a_sch.step()
        agent.c_sch.step()

        ep_history.append(reward_history)
        log_ep_text = {'epochs': epoch,
                       'time_step': agent.t,
                       'ep_reward': reward_history,
                       'entropy_acc_mean': entropy_acc_history / (t+1),
                       'entropy_ori_mean': entropy_ori_history / (t+1)}
        epochs.set_description(f'epochs:{epoch}, '
                               f'time_step:{agent.t}, '
                               f'reward:{reward_history:.1f}, '
                               f'entropy_acc:{log_ep_text["entropy_acc_mean"]:.1f}, '
                               f'entropy_ori:{log_ep_text["entropy_ori_mean"]:.1f}')
        # epoch数据写入log文件
        logger_ep.write2json(log_ep_text)

        # tensorboard logger
        tb_logger.add_scalar(tag='Reward/ep_reward',
                             scalar_value=reward_history,
                             global_step=epoch)
        tb_logger.add_scalar(tag='Reward/ep_entropy_acc',
                             scalar_value=log_ep_text["entropy_acc_mean"],
                             global_step=epoch)
        tb_logger.add_scalar(tag='Reward/ep_entropy_ori',
                             scalar_value=log_ep_text["entropy_ori_mean"],
                             global_step=epoch)
        tb_logger.add_image('Image/Trace',
                            np.array(trace_image),
                            global_step=epoch,
                            dataformats='HWC')

    agent.save_model(f'./log/{TIMESTAMP}/save_model_ep{epoch}.pth')
    env.close()


if __name__ == '__main__':
    config = parse_args()
    main(config)
