# -*- coding: utf-8 -*-
import csv
import json
import time
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os
import torch
from sys import platform

TIMESTAMP = str(round(time.time()))
KEYs_Train = ['epochs', 'time_step', 'ep_reward', 'entropy_mean']

FILE_NAME = '../vect_state/normal_vect_state/20221102'
# FILE_NAME = '../log/ppo_origin'

class log2json:
    def __init__(self, filename='train_log', type_json=True, log_path='log', logger_keys=None):
        self.root_path = os.getcwd()
        self.log_path = os.path.join(self.root_path, log_path, TIMESTAMP)

        # 创建当前训练log保存目录
        try:
            os.makedirs(self.log_path)
        except FileExistsError as e:
            print(e)

        filename = filename + TIMESTAMP
        if type_json:
            filename = os.path.join(self.log_path, filename + '.json').replace('\\', '/')
            self.fp = open(filename, 'w')
        else:
            filename = os.path.join(self.log_path, filename + '.csv')
            self.fp = open(filename, 'w', encoding='utf-8', newline='')
            self.csv_writer = csv.writer(self.fp)
            self.csv_keys = logger_keys
            self.csv_writer.writerow(self.csv_keys)

    def flush_write(self, string):
        self.fp.write(string + '\n')
        self.fp.flush()

    def write2json(self, log_dict):
        format_str = json.dumps(log_dict)
        self.flush_write(format_str)

    def write2csv(self, log_dict):
        format_str = list(log_dict.values())
        self.csv_writer.writerow(format_str)


class visualize_result:
    def __init__(self):
        sns.set_style("dark")
        sns.axes_style("darkgrid")
        sns.despine()
        sns.set_context("paper")

    @staticmethod
    def json2DataFrame(file_path):
        result_train = dict()
        # 创建train mode数据键值
        for key_str in KEYs_Train:
            result_train[key_str] = []
        # 文件读取
        with open(file_path, 'r') as fp:
            while True:
                json_line_str = fp.readline().rstrip('\n')
                if not json_line_str:
                    break
                _, temp_dict = json_line_extract(json_line_str)
                for key_iter in result_train.keys():
                    result_train[key_iter].append(temp_dict[key_iter])

        assert len(result_train['mode']) == len(result_train['ep_reward'])
        df_train = pd.DataFrame.from_dict(result_train)
        df_train.head()
        return df_train

    # @staticmethod
    def reward(self, logDataFrame):
        col = logDataFrame[0].shape[0]
        df_collect = pd.DataFrame(np.zeros((len(logDataFrame) * logDataFrame[0].shape[0], 3)),
                                  columns=['epochs', 'worker', 'ep_reward'])

        for idx, df_log in enumerate(logDataFrame):
            df_collect.iloc[idx*col:(idx+1)*col, 0] = df_log['epochs']
            df_collect['worker'][idx * col:(idx + 1) * col] = f'worker_{idx}'
            smooth_data = self._tensorboard_smoothing(df_log['ep_reward'], 0.0)
            df_collect.iloc[idx * col:(idx + 1) * col, 2] = smooth_data
        df_collect.head()

        # df_insert = pd.melt(df_insert, 'epochs', var_name='workers', value_name='ep_reward')
        # 将数据从长数据形式重塑
        df_collect = df_collect.pivot(index='epochs', columns=['worker'], values='ep_reward')
        df_collect.to_csv(os.path.join(FILE_NAME, 'smoothed_reward.csv'))

        # 不同worker累计回报可视化
        # self.sns_plot(df_collect, logDataFrame[0].shape[0])

    def uni_loss(self):
        pass

    def average_value(self):
        pass

    def _tensorboard_smoothing(self, values, smooth=0.9):
        # [0.81 0.9 1]. res[2] = (0.81 * values[0] + 0.9 * values[1] + values[2]) / 2.71
        norm_factor = smooth + 1
        x = values[0]
        res = [x]
        for i in range(1, len(values)):
            x = x * smooth + values[i]  # 指数衰减
            res.append(x / norm_factor)

            norm_factor *= smooth
            norm_factor += 1

        return res

    @staticmethod
    def sns_plot(df_collect, x_lim):
        sns.set_style('darkgrid')
        sns.set_context('paper')
        figure = sns.lineplot(data=df_collect, x='epochs', y='ep_reward', hue='worker')
        figure.set_xlim(0, x_lim)
        figure.set_ylim(-1500, 400)
        figure.set_yticks([-1500, -1000, -500, 0, 400])
        plt.xlabel('epoch', fontdict={'weight': 'bold',
                                      'size': 12})
        plt.ylabel('ep_reward', fontdict={'weight': 'bold', 'size': 12})
        # plt.subplots_adjust(left=0.2, bottom=0.2)
        # plt.title(y_trick, fontdict={'weight': 'bold',
        #                              'size': 20})
        # plt.legend(labels=['ppo ep reward'], loc='lower right', fontsize=10)
        plt.show()

    @staticmethod
    def dataframe_collect(file_name):
        df_train = pd.read_csv(file_name)
        df_train = df_train.drop(columns='Wall time')
        df_train.columns = ['epochs', 'ep_reward']
        return df_train


def dirs_creat():
    if platform.system() == 'windows':
        temp = os.getcwd()
        CURRENT_PATH = temp.replace('\\', '/')
    else:
        CURRENT_PATH = os.getcwd()
    ckpt_path = os.path.join(CURRENT_PATH, 'save_Model')
    log_path = os.path.join(CURRENT_PATH, 'log')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)


# 设置相同训练种子
def seed_torch(seed=2331):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)         # 为当前CPU 设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)        # 为当前的GPU 设置随机种子
        torch.cuda.manual_seed_all(seed)        # 当使用多块GPU 时，均设置随机种子
        torch.backends.cudnn.deterministic = True       # 设置每次返回的卷积算法是一致的
        torch.backends.cudnn.benchmark = True      # cuDNN使用的非确定性算法自动寻找最适合当前配置的高效算法，设置为False
        torch.backends.cudnn.enabled = True        # pytorch使用cuDNN加速


def json_line_extract(json_format_str):
    return json.loads(json_format_str)


if __name__ == '__main__':
    vg = visualize_result()
    df = []
    file_list = glob.glob(FILE_NAME+'/*.csv')
    for fp in file_list:
        path_csv_log = fp.replace("\\", "/")
        df.append(vg.dataframe_collect(path_csv_log))

    vg.reward(df)
