# -*-  coding=utf-8 -*-
# @Time : 2022/9/26 15:46
# @Author : Scotty1373
# @File : pandas_analy.py
# @Software : PyCharm
import pandas as pd
import numpy as np
import os

head = ["step", "Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]
file_path = '../log/1664271308_test/benchmark.csv'


def get_success(dataframe):
    success_rate = None
    fail_counter = np.zeros((6, 1))
    for i in range(dataframe.shape[0]):
        for j in range(1, dataframe.shape[1]):
            if dataframe.iloc[i, j] == 1000.:
                fail_counter[j-1, 0] += 1
    success_rate = 1 - fail_counter / dataframe.shape[0]
    return success_rate


def get_sort(dataframe):
    for col in head[1:]:
        dataframe.loc[:, col] = dataframe.loc[:, col].sort_values().reset_index(drop=True)
    # dataframe.loc['step'] = dataframe.index


if __name__ == '__main__':
    df = pd.read_csv(file_path)
    df.columns = head
    df.head()
    get_sort(df)
    plan_success_rate = get_success(df)
    for idx, area in enumerate(head[1:]):
        print(f'Env{area} successful route plan rate: {plan_success_rate[idx, 0]}')

    root_path = file_path.split('/')[:-1]
    root_path = '/'.join(root_path + ['benchmark_reset.csv'])
    df.to_csv(root_path)





