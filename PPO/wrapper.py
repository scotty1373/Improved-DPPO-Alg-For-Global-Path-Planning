# -*-  coding=utf-8 -*-
# @Time : 2022/6/29 10:59
# @Author : Scotty1373
# @File : wrapper.py
# @Software : PyCharm
import gym
from utils_tools.utils import RunningMeanStd, img_proc


class VecNormalize(gym.Wrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """
    def __init__(self, env, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, use_tf=False):
        super(VecNormalize, self).__init__(env)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)


class SkipEnvFrame(gym.Wrapper):
    def __init__(self, env, skip=3):
        super(SkipEnvFrame, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done, pixel, obs, info = None, None, None, None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            pixel = self.env.render()
            total_reward += reward
            if done:
                break
        return img_proc(pixel), obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
