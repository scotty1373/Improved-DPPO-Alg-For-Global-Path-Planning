# -*- coding: utf-8 -*-
import csv
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
from sys import platform

TIMESTAMP = str(round(time.time()))
KEYs_Train = ['mode', 'epochs', 'timestep', 'ep_reward']

FILE_NAME = ['../log/1647000521/train_log.json', '../log/1647002312/train_log.json']


class log2json:
    def __init__(self, filename='train_log', type_json=True, log_path='log', logger_keys=None):
        self.root_path = os.getcwd()
        self.log_path = os.path.join(self.root_path, log_path, TIMESTAMP)

        assert os.path.exists(os.path.join(self.root_path, log_path))

        # 创建当前训练log保存目录
        try:
            os.makedirs(self.log_path)
        except FileExistsError as e:
            print(e)

        if type_json:
            filename = os.path.join(self.log_path, filename + '.json')
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

    @staticmethod
    def reward(logDataFrame):
        for temp in logDataFrame:
            del temp['mode']
            del temp['timestep']

        df_insert = logDataFrame[0]
        for idx, df_log in enumerate(logDataFrame[1:]):
            df_insert.insert(idx, f'ep_reward_{idx}', df_log['ep_reward'])
        df_insert.head()
        df_insert = pd.melt(df_insert, 'epochs', var_name='workers', value_name='ep_reward')

        sns.lineplot(data=df_insert, x='epochs', y='ep_reward')
        # plt.xticks(np.linspace(0, df_train.index.values.max()*100, (df_train.index.values.max() + 1)))
        plt.xlabel('epochs')
        plt.show()

    def uni_loss(self):
        pass

    def average_value(self):
        pass


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
def seed_torch(seed=42):
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
    dict_data = json.loads(json_format_str)
    mode_log = True
    if dict_data['mode'] == 'val':
        mode_log = False
    return mode_log, dict_data


if __name__ == '__main__':
    vg = visualize_result()
    df = []
    for fp in FILE_NAME:
        df.append(vg.json2DataFrame(fp))

    vg.reward(df)



