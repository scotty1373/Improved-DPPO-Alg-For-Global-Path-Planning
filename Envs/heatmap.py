import math
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Box2D
from Box2D import (b2CircleShape, b2EdgeShape, b2PolygonShape)

ORG_SCALE = 16
REMAP_SACLE = 160
RATIO = REMAP_SACLE/ORG_SCALE


def heat_map_trans(vect, *, remap_scale=REMAP_SACLE, ratio=REMAP_SACLE / ORG_SCALE):
    """
    Mapping to xoy position
    The origin is at the bottom left
    :param vect: vect from Box2D position
    :param remap_scale: image size remapping 16 -> 160
                                            Box2D -> heatmap
    :param ratio: heatmap_size / Box2D_size
   """
    remap_vect = np.array((vect[0] * ratio + remap_scale / 2, vect[1] * ratio), dtype=np.uint16)
    return remap_vect


def heat_map_reverse(vect, *, ratio=REMAP_SACLE/ORG_SCALE):
    remap_vect = np.array(((vect[0] - (REMAP_SACLE / 2)) / ratio,
                           (vect[1] / ratio)), dtype=np.float32)
    return remap_vect


def get_dist(pointa, pointb):
    return np.sqrt(np.square(pointa[0] - pointb[0]) + np.square(pointa[1] - pointb[1]))


def normalize(array):
    assert isinstance(array, np.ndarray)
    return (array - array.min()) / (array.max() - array.min())


class HeatMap:
    def __init__(self, bound_list, *, positive_reward=None, ground_size=(REMAP_SACLE, REMAP_SACLE)):
        self.size = ground_size
        self._init(bound_list)
        self.barr_reward = -20
        self.ground_pean = -20
        self.reach_reward = 50
        if positive_reward is not None:
            self.positive = True
        else:
            self.positive = None

    def _init(self, bound_list):
        # 原始mat
        self.mat = self._generate_mat(self.size)
        self.ones_mat = self._generate_mat(self.size)
        # 障碍mat
        self.bl, self.ra, self.ga = self._GenerateBarrierInfo(bound_list)

    @staticmethod
    def _generate_mat(mat_size, ones_only=False):
        """
        :param mat_size: tuple (row, col)
        :return: 2D ndarray
        """
        if ones_only:
            return np.ones(mat_size)
        else:
            return np.zeros(mat_size)

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
                    # box2d提供的横纵坐标相反
                    barrier_list['position'].append(heat_map_trans(trans*f.shape.pos))
                    barrier_list['radius'].append(f.shape.radius * RATIO)
                # 边界控制参数
                # [todo] chain边界会导致多个fixture创建
                elif type(f.shape) == b2EdgeShape:
                    all_vertices = [heat_map_trans(trans * vertice) for vertice in f.shape.all_vertices]
                    ground_area['xyxy'] = [all_vertices[0], all_vertices[2]]
                    ground_area['scope'] = REMAP_SACLE//2*0.9
                    break
                elif type(f.shape) == b2PolygonShape:
                    if hasattr(obj, 'color'):
                        reach_area['position'] = heat_map_trans(obj.position)
                        reach_area['radius'] = math.sqrt(2)/2 * RATIO

        return barrier_list, reach_area, ground_area

    @property
    def rewardCal(self):
        """
        基于障碍物的reward计算`
        :return: nd.ndarray matrix
        """
        heat_mat_collect = self.mat.copy()
        barr_num = len(self.bl['position'])
        ratio = 2.5
        for idx_barr in range(barr_num):
            heat_mat = self.mat.copy()
            # 判断输入障碍物坐标非法
            assert isinstance(self.bl['position'][idx_barr][0], np.uint16)
            assert isinstance(self.bl['position'][idx_barr][1], np.uint16)
            if self.bl['position'][idx_barr][0] > self.size[0]:
                raise ValueError
            if self.bl['position'][idx_barr][1] > self.size[1]:
                raise ValueError

            for row_offset in range(self.size[0]):
                for col_offset in range(self.size[1]):
                    # 跳过自身位置，防止divide by zero
                    dist = get_dist(self.bl['position'][idx_barr], (row_offset, col_offset))
                    if dist <= self.bl['radius'][idx_barr]:
                        heat_mat[row_offset, col_offset] = self.barr_reward * (1.5 - math.log(self.bl['radius'][idx_barr]*0.05))
                        continue
                    elif self.bl['radius'][idx_barr] < dist <= self.bl['radius'][idx_barr] * ratio:
                        heat_mat[row_offset, col_offset] = self.barr_reward * (1.5 - math.log(dist - self.bl['radius'][idx_barr]*0.95))
            heat_index = heat_mat != 0
            heat_mat[heat_index] = (heat_mat[heat_mat != 0] - heat_mat[heat_mat != 0].min()) / (heat_mat[heat_mat != 0].max() - heat_mat[heat_mat != 0].min())
            heat_mat[heat_index] = 1 - heat_mat[heat_index]
            heat_mat_collect += heat_mat
        return -heat_mat_collect

    @property
    def rewardCal4TraditionalMethod(self):
        """
        基于障碍物的reward计算
        :return: np.ndarray matrix
        """
        heat_mat = np.ones_like(self.mat)
        barr_num = len(self.bl['position'])
        ratio = 1
        for idx_barr in range(barr_num):
            # 判断输入障碍物坐标非法
            assert isinstance(self.bl['position'][idx_barr][0], np.uint16)
            assert isinstance(self.bl['position'][idx_barr][1], np.uint16)
            if self.bl['position'][idx_barr][0] > self.size[0]:
                raise ValueError
            if self.bl['position'][idx_barr][1] > self.size[1]:
                raise ValueError

            for row_offset in range(self.size[0]):
                for col_offset in range(self.size[1]):
                    # 跳过自身位置，防止divide by zero
                    dist = get_dist(self.bl['position'][idx_barr], (row_offset, col_offset))
                    if dist <= self.bl['radius'][idx_barr] * ratio:
                        heat_mat[row_offset, col_offset] = 0
        # heat_mat = np.flipud(heat_mat.T)
        return heat_mat

    @property
    def ground_rewardCal(self):
        """
        :return: 2D-array
        """
        _mat = self.mat.copy()
        center = (self.size[0]//2-1, self.size[1]//2-1)
        for row_offset in range(self.size[0]):
            for col_offset in range(self.size[1]):
                dist = get_dist(center, (row_offset, col_offset))
                ratio = 0.95
                if dist <= self.size[0]//2 * ratio:
                    """测试修改"""
                    _mat[row_offset, col_offset] = self.ground_pean * (1 / (-math.log(self.size[0]//2 * ratio) + 5))
                else:
                    _mat[row_offset, col_offset] = self.ground_pean * (1 / (-math.log(dist) + 5)) * 2
        return normalize(_mat) - 1

    @property
    def ground_rewardCal_redesign(self):
        """
        :return: 2D-array
        """
        _mat = self.mat.copy()
        limit = 0.0625

        def is_inrange(num):
            return num if not REMAP_SACLE * limit <= num < REMAP_SACLE * (1 - limit) else None

        # filter 会过滤掉返回的0和其他在Python中为None的类型数据, we start from 1
        linspace = [0] + list(filter(is_inrange, range(1, self.size[0])))
        linspace = np.array(linspace)
        limit_lower = self.size[0] * limit
        limit_upper = self.size[0] * (1-limit)
        yaxis_bias = 3
        for row_offset in range(self.size[0]):
            for col_offset in range(self.size[0]):
                if row_offset in linspace:
                    if row_offset >= limit_upper:
                        if col_offset < limit_lower or col_offset > limit_upper:
                            target_point = (limit_lower if row_offset<limit_lower else limit_upper,
                                            limit_lower if col_offset<limit_lower else limit_upper)
                            dist = get_dist(target_point, (row_offset, col_offset))
                            _mat[row_offset, col_offset] = \
                                self.ground_pean * (1 / (-math.log(dist + 1) + yaxis_bias))
                        else:
                            _mat[row_offset, col_offset] = \
                                self.ground_pean*(1/(-math.log(row_offset-limit_upper+1)+yaxis_bias))
                    else:
                        if col_offset < limit_lower or col_offset > limit_upper:
                            target_point = (limit_lower if row_offset<limit_lower else limit_upper,
                                            limit_lower if col_offset<limit_lower else limit_upper)
                            dist = get_dist(target_point, (row_offset, col_offset))
                            _mat[row_offset, col_offset] = \
                                self.ground_pean * (1 / (-math.log(dist + 1) + yaxis_bias))
                        else:
                            _mat[row_offset, col_offset] = \
                                self.ground_pean*(1/(-math.log(-(row_offset-limit_lower)+1)+yaxis_bias))
                if col_offset in linspace:
                    if col_offset >= limit_upper:
                        if row_offset < limit_lower or row_offset > limit_upper:
                            target_point = (limit_lower if row_offset<limit_lower else limit_upper,
                                            limit_lower if col_offset<limit_lower else limit_upper)
                            dist = get_dist(target_point, (row_offset, col_offset))
                            _mat[row_offset, col_offset] = \
                                self.ground_pean * (1 / (-math.log(dist + 1) + yaxis_bias))
                        else:
                            _mat[row_offset, col_offset] = \
                                self.ground_pean*(1/(-math.log(col_offset-limit_upper+1)+yaxis_bias))
                    else:
                        if row_offset < limit_lower or row_offset > limit_upper:
                            target_point = (limit_lower if row_offset < limit_lower else limit_upper,
                                            limit_lower if col_offset < limit_lower else limit_upper)
                            dist = get_dist(target_point, (row_offset, col_offset))
                            _mat[row_offset, col_offset] = \
                                self.ground_pean * (1 / (-math.log(dist + 1) + yaxis_bias))
                        else:
                            _mat[row_offset, col_offset] = \
                                self.ground_pean*(1/(-math.log(-(col_offset-limit_lower)+1)+yaxis_bias))
        return normalize(_mat) - 1

    @property
    def reach_rewardCal(self):
        """
        :return: np.ndarray matrix
        """
        _mat = self.mat.copy()
        for row_offset in range(self.size[0]):
            for col_offset in range(self.size[1]):
                dist = get_dist(self.ra['position'], (row_offset, col_offset))
                ratio = 0.5
                if dist <= self.ra['radius']:
                    _mat[row_offset, col_offset] = self.reach_reward * (6 - math.log(self.ra['radius']*ratio))
                    continue
                else:
                    _mat[row_offset, col_offset] = self.reach_reward * (6 - math.log(dist-self.ra['radius']*(1-ratio)))
        return normalize(_mat) - 1 if self.positive is None else normalize(_mat)


if __name__ == '__main__':
    env = HeatMap(3)
    mat = env.mat.copy()
    mat += env.ground_rewardCal(env.mat.copy())
    for barr in env.barr_list:
        mat += env.rewardCal(env.mat.copy(), barr)
        # fig, axes = plt.subplots(2, 1)
        # ax1 = axes[0]
        # ax2 = axes[1]
        # sns.heatmap(mat, annot=False, ax=ax1)
        # ax2.imshow(env.barr_mat, cmap='gray')
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, axes = plt.subplots(1, 1)
        sns.heatmap(_mat, annot=False, ax=axes)
        plt.show()

    time.time()
