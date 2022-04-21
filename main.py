# -*- coding: utf-8 -*-
import numpy as np

from Envs.sea_env import RoutePlan
from PPO.PPO import PPO
from utils_tools.common import log2json
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='PPO config option')
    parser.add_argument('--epochs',
                        help='Training epoch',
                        default=50)
    parser.add_argument('--pre_train',
                        help='Pretrained?',
                        default=False)
    parser.add_argument('--checkpoint',
                        help='If pre_traine is True, this option is pretrained ckpt path',
                        default=None)
    parser.add_argument('--max_timestep',
                        help='Maximum time step in a single epoch',
                        default=2000)
    args = parser.parse_args()
    return args


def main(args):
    args = args
    env = RoutePlan(barrier_num=5)
    agent = PPO(state_dim=7+11, action_dim=2, batch_size=16)

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
        step = tqdm(range(args.max_timestep))
        for t in step:
            env.render()
            acc, ori, mean = agent.get_action(obs)
            obs_t1, reward, done, _ = env.step(np.concatenate((acc[..., None, None], ori[..., None, None]), axis=1).squeeze())
            agent.state_store_memory(obs, acc, ori, reward, mean[..., 0], mean[..., 1])

            if (t+1) % agent.batch_size == 0 or (done and t%agent.batch_size>5):
                state, action_acc, action_ori, reward_nstep, _, _ = zip(*agent.memory)
                state = np.stack(state, axis=0)
                action_acc = np.stack(action_acc, axis=0)
                action_ori = np.stack(action_ori, axis=0)
                reward_nstep = np.stack(reward_nstep, axis=0)
                discount_reward = agent.decayed_reward(obs_t1, reward_nstep)
                agent.update(state, action_acc, action_ori, discount_reward)
                agent.memory.clear()

            if done:
                break

            # 记录timestep, reward＿sum
            agent.t += 1
            obs = obs_t1
            reward_history += reward

        ep_history.append(reward_history)
    env.close()


if __name__ == '__main__':
    config = parse_args()
    main(config)
