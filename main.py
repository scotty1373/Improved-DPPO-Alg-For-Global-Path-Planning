# -*- coding: utf-8 -*-
import numpy as np

from Envs.sea_env import RoutePlan
from PPO.PPO import PPO
from utils_tools.common import log2json
from tqdm import tqdm
import argparse

TIME_BOUNDARY = 500

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
                        default=500)
    args = parser.parse_args()
    return args


def main(args):
    args = args
    env = RoutePlan(barrier_num=3, seed=12)
    env.seed(13)
    agent = PPO(state_dim=6+11, action_dim=2, batch_size=16)

    # log 初始化
    # log = log2json()

    # 预训练？
    if args.pre_train:
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
            agent.load_model(checkpoint)

    ep_history = []
    epochs = tqdm(range(args.epochs))

    for epoch in epochs:
        reward_history = 0
        obs, _, done, _ = env.reset()
        step = tqdm(range(args.max_timestep*5))
        for t in step:
            env.render()
            # if t%5==0:
            act, logprob, dist = agent.get_action(obs)

            obs_t1, reward, done, _ = env.step(act)
            agent.state_store_memory(obs, act, reward, logprob)

            if (agent.t + 1) % agent.batch_size == 0 or (done and agent.t % agent.batch_size > 5):
                state, action, reward_nstep, logprob_nstep = zip(*agent.memory)
                state = np.stack(state, axis=0)
                action = np.stack(action, axis=0)
                reward_nstep = np.stack(reward_nstep, axis=0)
                logprob_nstep = np.stack(logprob_nstep, axis=0)
                discount_reward = agent.decayed_reward(obs_t1, reward_nstep)
                agent.update(state, action, logprob_nstep, discount_reward)
                agent.memory.clear()
            step.set_description(f'epochs:{epoch}, '
                                 f'time_step:{agent.t}, '
                                 f'reward:{reward:.1f}, '
                                 f'entropy: {dist.entropy().numpy().sum().item():.1f}, '
                                 f'acc:{act[0].item():.1f}, '
                                 f'ori:{act[1].item():.1f}, '
                                 f'ang_vel:{env.ship.angularVelocity:.1f}, '
                                 f'actor_loss:{agent.history_actor:.1f}, '
                                 f'critic_loss:{agent.history_critic:.1f}')
            if done:
                break
            # 记录timestep, reward＿sum
            agent.t += 1
            obs = obs_t1
            reward_history += reward

            # else:
            #     env.step(np.zeros(2,))

            if t + 1 % (args.max_timestep*5) == 0:
                break

        ep_history.append(reward_history)
    env.close()


if __name__ == '__main__':
    config = parse_args()
    main(config)
