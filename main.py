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

    # Iter log初始化
    logger_iter = log2json(filename='train_log_iter', type_json=True)
    # epoch log初始化
    logger_ep = log2json(filename='train_log_ep', type_json=True)

    # 预训练？
    if args.pre_train:
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
            agent.load_model(checkpoint)

    ep_history = []

    epochs = tqdm(range(args.epochs), leave=False, position=0, colour='green')

    for epoch in epochs:
        reward_history = 0
        entropy_history = 0
        obs = env.reset()
        step = tqdm(range(args.max_timestep*5), leave=False, position=1, colour='red')
        for t in step:
            env.render()
            # if t%5==0:
            act, logprob, dist = agent.get_action(obs)

            obs_t1, reward, done, _ = env.step(act)
            if not args.pre_train:
                agent.state_store_memory(obs, act, reward, logprob)

                if (t+1) % agent.batch_size == 0 or (done and (t+1) % agent.batch_size > 5):
                    state, action, reward_nstep, logprob_nstep = zip(*agent.memory)
                    state = np.stack(state, axis=0)
                    action = np.stack(action, axis=0)
                    logprob_nstep = np.stack(logprob_nstep, axis=0)
                    reward_nstep = np.stack(reward_nstep, axis=0)
                    discount_reward = agent.decayed_reward(obs_t1, reward_nstep)
                    agent.update(state, action, logprob_nstep, discount_reward)
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

            if t + 1 % (args.max_timestep*5) == 0:
                break

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
    env.close()


if __name__ == '__main__':
    config = parse_args()
    main(config)
