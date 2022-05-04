# -*-  coding=utf-8 -*-
# @Time : 2022/4/23 16:47
# @Author : Scotty1373
# @File : temp.py
# @Software : PyCharm
import numpy as np
from scipy import signal

if __name__ == '__main__':
    td_error = np.ones((16, 1))
    gae_advantage = np.zeros((16, 1))

    for idx, td in enumerate(td_error[::-1]):
        temp = 0
        for adv_idx, weight in enumerate(range(idx, -1, -1)):
            temp += gae_advantage[adv_idx, 0] * ((0.95 * 0.99) ** weight)
        gae_advantage[idx, ...] = td + temp

    gae = signal.lfilter([1], [1, -0.99 * 0.95], td_error[::-1], axis=0)[::-1]