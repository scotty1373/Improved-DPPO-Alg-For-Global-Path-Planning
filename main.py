# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from Envs.sea_env_without_orient import RoutePlan
from PPO.PPO import PPO
from utils_tools.common import log2json, dirs_creat, TIMESTAMP
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import argparse

TIME_BOUNDARY = 500


def parse_args():
    parser = argparse.ArgumentParser(
        description='PPO config option')
    parser.add_argument('--epochs',
                        help='Training epoch',
                        default=300)
    parser.add_argument('--pre_train',
                        help='Pretrained?',
                        default=False)
    parser.add_argument('--checkpoint',
                        help='If pre_traine is True, this option is pretrained ckpt path',
                        default=None)
    parser.add_argument('--max_timestep',
                        help='Maximum time step in a single epoch',
                        default=1000)
    parser.add_argument('--seed',
                        help='environment initialization seed',
                        default=42)
    parser.add_argument('--batch_size',
                        help='training batch size',
                        default=16)
    parser.add_argument('--frame_skipping',
                        help='random walk frame skipping',
                        default=2)
    parser.add_argument('--frame_overlay',
                        help='data frame overlay',
                        default=3)
    parser.add_argument('--state_length',
                        help='state data vector length',
                        default=5+24*2)
    parser.add_argument('--pixel_state',
                        help='Image-Based Status',
                        default=False)
    args = parser.parse_args()
    return args


# 数据帧叠加
def state_frame_overlay(new_state, old_state, frame_num):
    new_frame_overlay = np.concatenate((new_state.reshape(1, -1),
                                        old_state.reshape(frame_num, -1)[:(frame_num - 1), ...]),
                                       axis=0).reshape(-1)
    return new_frame_overlay


# 基于图像的数据帧叠加
def pixel_based(new_state, old_state, frame_num):
    pass


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
    env.unwrapped
    assert isinstance(args.batch_size, int)
    # agent = PPO(state_dim=3*(7+24), action_dim=2, batch_size=args.batch_size)
    agent = PPO(state_dim=args.frame_overlay * args.state_length,
                action_dim=2,
                batch_size=args.batch_size)

    # Iter log初始化
    logger_iter = log2json(filename='train_log_iter', type_json=True)
    # epoch log初始化
    logger_ep = log2json(filename='train_log_ep', type_json=True)
    # tensorboard初始化
    tb_logger = SummaryWriter(log_dir=f"./log/{TIMESTAMP}", flush_secs=120)
    fig, ax1 = plt.subplots(1, 1)
    sns.heatmap(env.heat_map, ax=ax1)
    fig.suptitle('reward shaping heatmap')
    tb_logger.add_figure('figure', fig)

    # 是否从预训练结果中载入ckpt
    if args.pre_train:
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
            agent.load_model(checkpoint)

    ep_history = []
    epochs = tqdm(range(args.epochs), leave=False, position=0, colour='green')

    for epoch in epochs:
        reward_history = 0
        entropy_history = 0
        obs, _, done, _ = env.reset()
        # obs = np.stack((obs, obs, obs), axis=0).reshape(-1)
        '''利用广播机制初始化state帧叠加结构，不使用stack重复对数组进行操作'''
        obs = (np.ones((args.frame_overlay, args.state_length)) * obs).reshape(-1)
        step = tqdm(range(1, args.max_timestep*args.frame_skipping), leave=False, position=1, colour='red')
        for t in step:
            # 是否进行可视化渲染
            env.render()
            if t % args.frame_skipping == 0:
                act, logprob, dist = agent.get_action(obs)
                # 环境交互
                obs_t1, reward, done, _ = env.step(act, t)
                if args.frame_overlay == 1:
                    pass
                else:
                    obs_t1 = state_frame_overlay(obs_t1, obs, args.frame_overlay)
                # 达到maxstep次数之后给予惩罚
                if (t + args.frame_skipping) % args.max_timestep == 0:
                    done = True
                    reward = -10

                if not args.pre_train:

                    # 状态存储
                    agent.state_store_memory(obs, act, reward, logprob)

                    if t % (args.frame_skipping*agent.batch_size) == 0 or (done and t % (agent.batch_size*args.frame_skipping) > 5):
                        state, action, reward_nstep, logprob_nstep = zip(*agent.memory)
                        state = np.stack(state, axis=0)
                        action = np.stack(action, axis=0)
                        reward_nstep = np.stack(reward_nstep, axis=0)
                        logprob_nstep = np.stack(logprob_nstep, axis=0)
                        # 动作价值计算
                        discount_reward = agent.decayed_reward(obs_t1, reward_nstep)

                        # 计算gae advantage
                        with torch.no_grad():
                            last_frame = torch.Tensor(obs_t1)
                            last_val = agent.v(last_frame)
                        # 策略网络价值网络更新
                        agent.update(state, action, logprob_nstep, discount_reward, reward_nstep, last_val, done)
                        # 清空存储池
                        agent.memory.clear()
                entropy = dist.entropy().numpy().sum().item()
                log_text = {'epochs': epoch,
                            'time_step': agent.t,
                            'reward': reward,
                            'entropy': entropy,
                            'acc': act[0].item(),
                            'ori': act[1].item(),
                            'actor_loss': agent.history_actor,
                            'critic_loss': agent.history_critic}
                step.set_description(f'epochs:{epoch}, '
                                     f'time_step:{agent.t}, '
                                     f'reward:{reward:.1f}, '
                                     f'entropy: {log_text["entropy"]:.1f}, '
                                     f'acc:{log_text["acc"]:.1f}, '
                                     f'ori:{log_text["ori"]:.1f}, '
                                     f'lr:{agent.a_opt.state_dict()["param_groups"][0]["lr"]:.5f}, '
                                     f'ang_vel:{env.ship.angularVelocity:.1f}, '
                                     f'actor_loss:{agent.history_actor:.1f}, '
                                     f'critic_loss:{agent.history_critic:.1f}')
                # iter数据写入log文件
                logger_iter.write2json(log_text)

                # 记录timestep, reward＿sum
                agent.t += 1
                obs = obs_t1
                reward_history += reward
                entropy_history += entropy

                if done:
                    break

            else:
                env.step(np.zeros(2,), t)

            if (t + 1) % (args.max_timestep * args.frame_skipping) == 0:
                break

        # lr_Scheduler
        agent.a_sch.step()
        agent.c_sch.step()

        ep_history.append(reward_history)
        log_ep_text = {'epochs': epoch,
                       'time_step': agent.t,
                       'ep_reward': reward_history,
                       'entropy_mean': entropy_history / (t+1)}
        epochs.set_description(f'epochs:{epoch}, '
                               f'time_step:{agent.t}, '
                               f'reward:{reward_history:.1f}, '
                               f'entropy:{log_ep_text["entropy_mean"]:.1f}')
        # epoch数据写入log文件
        logger_ep.write2json(log_ep_text)

        # tensorboard logger
        tb_logger.add_scalar(tag='Loss/ep_reward',
                             scalar_value=reward_history,
                             global_step=epoch)
        tb_logger.add_scalar(tag='Loss/ep_entropy',
                             scalar_value=log_ep_text["entropy_mean"],
                             global_step=epoch)

    agent.save_model(f'./save_model/save_model_ep{epoch}.pth')
    env.close()


if __name__ == '__main__':
    config = parse_args()
    main(config)
