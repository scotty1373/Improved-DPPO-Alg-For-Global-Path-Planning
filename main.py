# -*- coding: utf-8 -*-
import numpy as np

from Envs.sea_env import RoutePlan
from PPO.PPO import PPO
from utils_tools.common import log2json
from tqdm import tqdm
import argparse
import gym

TIME_BOUNDARY = 256

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
                        help='If pre_train is True, this option is pretrained ckpt path',
                        default=None)
    parser.add_argument('--max_timestep',
                        help='Maximum time step in a single epoch',
                        default=500)
    args = parser.parse_args()
    return args


def main(args):
    args = args
    env = gym.make('LunarLanderContinuous-v2')
    env.unwrapped
    agent = PPO(state_dim=8, action_dim=2, batch_size=16)

    # log 初始化
    # log = log2json()

    # 预训练？
    if args.pre_train:
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
            agent.load_model(checkpoint)

    ep_history = []
    epochs = tqdm(range(args.epochs), leave=False, position=0, colour='green')

    for epoch in epochs:
        reward_history = 0
        obs = env.reset()
        step = tqdm(range(args.max_timestep*5), leave=False, position=1, colour='red')
        for t in step:
            env.render()
            # if t%5==0:
            act, logprob, dist = agent.get_action(obs)

            obs_t1, reward, done, _ = env.step(act)
            if not args.pre_train:
                agent.state_store_memory(obs, act, reward, logprob)

                if (agent.t + 1) % agent.batch_size == 0 or (done and agent.t % agent.batch_size > 5):
                    state, action, reward_nstep, logprob_nstep = zip(*agent.memory)
                    state = np.stack(state, axis=0)
                    action = np.stack(action, axis=0)
                    logprob_nstep = np.stack(logprob_nstep, axis=0)
                    reward_nstep = np.stack(reward_nstep, axis=0)
                    discount_reward = agent.decayed_reward(obs_t1, reward_nstep)
                    agent.update(state, action, logprob_nstep, discount_reward)
                    agent.memory.clear()
            step.set_description(f'epochs:{epoch}, '
                                 f'time_step:{agent.t}, '
                                 f'reward:{reward:.1f}, '
                                 f'entropy: {dist.entropy().numpy().sum().item():.1f}, '
                                 f'acc:{act[0].item():.1f}, '
                                 f'ori:{act[1].item():.1f}')
            if done:
                break
            # 记录timestep, reward＿sum
            agent.t += 1
            obs = obs_t1
            reward_history += reward

            if t + 1 % (args.max_timestep*5) == 0:
                break

        ep_history.append(reward_history)
    env.close()


if __name__ == '__main__':
    config = parse_args()
    main(config)
