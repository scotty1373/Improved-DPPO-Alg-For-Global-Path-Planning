# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 0:30
# @Author  : Scotty
# @FileName: test.py
# @Software: PyCharm
import time

import torch.multiprocessing as mp
import torch


class worker(mp.Process):
    def __init__(self, workerID, datashare, pipe):
        super(worker, self).__init__()
        self.worker = workerID
        self.share = datashare
        self.pipe = pipe

    def run(self):
        for i in range(10):
            pixel_state = torch.randn((16, 3, 80, 80))
            vect_state = torch.randn((16, 3 * 2))

            temp_pixel = self.share['pixel']
            temp_pixel.append(pixel_state)

            temp_vect = self.share['vect']
            temp_vect.append(vect_state)

            self.share.update({'pixel': temp_pixel, 'vect': temp_vect})

            self.pipe.send((pixel_state, vect_state))
        # self.pipe.close()


class buffer:
    def __init__(self, dict_share):
        self.pixel_buffer = []
        self.vect_buffer = []
        self.dict_share = dict_share

    def get_dict(self):
        self.dict_share['pixel'] = self.pixel_buffer
        self.dict_share['vect'] = self.vect_buffer


if __name__ == '__main__':
    # queue initialize
    pixel_buffer = mp.Queue()
    vect_buffer = mp.Queue()
    action_buffer = mp.Queue()
    log_prob_buffer = mp.Queue()
    reward_buffer = mp.Queue()
    done_buffer = mp.Queue()

    data_share = mp.Manager().dict()
    ppo_buffer = buffer(data_share)
    ppo_buffer.get_dict()

    pipe_r, pipe_w = zip(*[mp.Pipe() for _ in range(3)])

    worker_list = [worker(i, data_share, pipe_r[i]) for i in range(3)]
    data = [[] for _ in range(3)]
    [w.start() for w in worker_list]

    for i in range(3):
        for j in range(10):
            data[i].append(pipe_w[i].recv())

    time.time()








