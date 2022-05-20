import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Box2D
from Box2D import (b2CircleShape, b2EdgeShape, b2PolygonShape)

ORG_SCALE = 16
REMAP_SACLE = 40
RATIO = REMAP_SACLE/ORG_SCALE


def heat_map_trans(vect, *, ratio=REMAP_SACLE/ORG_SCALE):
    remap_vect = np.array((vect[0] * ratio + (REMAP_SACLE / 2), (-vect[1] * ratio) + REMAP_SACLE), dtype=np.uint8)
    return remap_vect


def get_dist(pointa, pointb):
    return np.sqrt(np.square(pointa[0] - pointb[0]) + np.square(pointa[1] - pointb[1]))


def normalize(array):
    assert isinstance(array, np.ndarray)
    return (array - array.min()) / (array.max() - array.min())


class HeatMap:
    def __init__(self, bound_list):
        self.size = (REMAP_SACLE, REMAP_SACLE)
        self._init(bound_list)
        self.barr_reward = -20
        self.ground_pean = -20
        self.reach_reward = 50

    def _init(self, bound_list):
        # 原始mat
        self.mat = self._generate_mat(self.size)
        # 障碍mat
        self.bl, self.ra, self.ga = self._GenerateBarrierInfo(bound_list)

    @staticmethod
    def _generate_mat(mat_size):
        """
        :param mat_size: tuple (row, col)
        :return: 2D ndarray
        """
        _mat = np.zeros(mat_size)
        return _mat

    def _GenerateBarrierInfo(self, bound_list):
        """
        根据障碍物、边界范围、抵达点创建障碍物索引
        障碍物所需构建heatmap参数为：
            1.world_position
            2.radius
        边界所需构建heatmap参数为：
            1.边界world_position
            2.开始计算reward的边界radius范围
        抵达点所需构建heatmap参数为：
            1.world_position
            2.radius
        [todo: 抵达点位置周围计算reward与边界reward相重合时应怎样计算]
        :param _mat: numpy ndarray, dim == 2
        :return:
        """
        barrier_list = {'position': [],
                        'radius': []}
        reach_area = dict()
        ground_area = dict()

        for idx, obj in enumerate(bound_list):
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) == b2CircleShape:
                    # 障碍物参数
                    if hasattr(obj, 'color'):
                        reach_area['position'] = heat_map_trans(obj.position)[[1, 0]]
                        reach_area['radius'] = f.shape.radius * RATIO
                    # 抵达点参数
                    else:
                        # box2d提供的横纵坐标相反
                        barrier_list['position'].append(heat_map_trans(trans*f.shape.pos)[[1, 0]])
                        barrier_list['radius'].append(f.shape.radius * RATIO)
                # 边界控制参数
                # [todo] chain边界会导致多个fixture创建
                elif type(f.shape) == b2EdgeShape:
                    all_vertices = [heat_map_trans(trans * vertice) for vertice in f.shape.all_vertices]
                    ground_area['xyxy'] = [all_vertices[0], all_vertices[2]]
                    ground_area['scope'] = REMAP_SACLE//2*0.9
                    break
        return barrier_list, reach_area, ground_area

    def rewardCal(self, barr_list):
        """
        基于障碍物的reward计算

        :param _mat: numpy ndarray
        :param barr_list: position of barrier
        :return:
        """
        heat_mat_collect = self.mat.copy()
        heat_mat = self.mat.copy()
        barr_num = len(barr_list['position'])
        for idx_barr in range(barr_num):
            # 判断输入障碍物坐标非法
            assert isinstance(barr_list['position'][idx_barr][0], np.uint8)
            assert isinstance(barr_list['position'][idx_barr][1], np.uint8)
            if barr_list['position'][idx_barr][0] > self.size[0]:
                raise ValueError
            if barr_list['position'][idx_barr][1] > self.size[1]:
                raise ValueError

            for row_offset in range(self.size[0]):
                for col_offset in range(self.size[1]):
                    # 跳过自身位置，防止divide by zero
                    dist = get_dist(barr_list['position'][idx_barr], (row_offset, col_offset))
                    if dist <= barr_list['radius'][idx_barr]:
                        heat_mat[row_offset, col_offset] = self.barr_reward * (1.5 - math.log(barr_list['radius'][idx_barr]*0.05))
                        continue
                    elif barr_list['radius'][idx_barr] < dist <= barr_list['radius'][idx_barr] * 1.75:
                        heat_mat[row_offset, col_offset] = self.barr_reward * (1.5 - math.log(dist - barr_list['radius'][idx_barr]*0.95))
                    else:
                        pass
            heat_index = heat_mat != 0
            heat_mat[heat_index] = (heat_mat[heat_mat != 0] - heat_mat[heat_mat != 0].min()) / (heat_mat[heat_mat != 0].max() - heat_mat[heat_mat != 0].min())
            heat_mat[heat_index] = 1 - heat_mat[heat_index]
            heat_mat_collect += heat_mat
        return -heat_mat_collect

    @property
    def ground_rewardCal(self):
        """
        :return: 2D-array
        """
        _mat = self.mat.copy()
        center = (self.size[0]//2, self.size[1]//2)
        for row_offset in range(self.size[0]):
            for col_offset in range(self.size[1]):
                dist = get_dist(center, (row_offset, col_offset))
                if dist <= self.size[0]//2 * 0.4:
                    """测试修改"""
                    _mat[row_offset, col_offset] = self.ground_pean * (1 / (-math.log(self.size[0]//2 * 0.4) + 3.8))
                else:
                    _mat[row_offset, col_offset] = self.ground_pean * (1 / (-math.log(dist) + 3.8))
        return (- 1 + normalize(_mat)) * 2

    def reach_rewardCal(self, reachinfo):
        """
        :param reachinfo: numpy ndarray
        :return:
        """
        _mat = self.mat.copy()
        for row_offset in range(self.size[0]):
            for col_offset in range(self.size[1]):
                dist = get_dist(reachinfo['position'], (row_offset, col_offset))
                if dist <= reachinfo['radius']:
                    _mat[row_offset, col_offset] = self.reach_reward * (8 - math.log(reachinfo['radius']*0.2))
                    continue
                else:
                    _mat[row_offset, col_offset] = self.reach_reward * (8 - math.log(dist - reachinfo['radius']*0.8))
        return normalize(_mat)


if __name__ == '__main__':
    # env = HeatMap()
    mat = env.mat.copy()
    mat += env.ground_rewardCal(env.mat.copy())
    for barr in env.barr_list:
        mat += env.rewardCal(env.mat.copy(), barr)
        # fig, axes = plt.subplots(2, 1)
        # ax1 = axes[0]
        # ax2 = axes[1]
        # sns.heatmap(mat, annot=False, ax=ax1)
        # ax2.imshow(env.barr_mat, cmap='gray')
        fig, axes = plt.subplots(1, 1)
        sns.heatmap(heat_mat_collect, annot=False, ax=axes)
        plt.show()

    time.time()
