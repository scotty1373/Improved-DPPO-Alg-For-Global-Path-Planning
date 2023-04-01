# -*-  coding=utf-8 -*-
# @Time : 2022/4/10 10:46
# @Author : Scotty1373
# @File : sea_env.py
# @Software : PyCharm
import copy
import math
import random
import time

import numpy as np
import keyboard

import Box2D
from Box2D import (b2CircleShape, b2FixtureDef,
                   b2PolygonShape, b2ContactListener,
                   b2Distance, b2RayCastCallback,
                   b2Vec2, b2_pi, b2Dot)
"""
b2FixtureDef: 添加物体材质
b2PolygonShape： 多边形构造函数，初始化数据
b2CircleShape: 圆形构造函数
b2ContactListener：碰撞检测监听器
"""

import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from .heatmap import HeatMap, heat_map_trans, normalize
from utils_tools.utils import img_proc

SCALE = 30
FPS = 60

VIEWPORT_W = 480
VIEWPORT_H = 480

INITIAL_RANDOM = 20
MAIN_ENGINE_POWER = 70
MAIN_ORIENT_POWER = b2_pi
SIDE_ENGINE_POWER = 5

#           Background           PolyLine
PANEL = [(0.19, 0.72, 0.87), (0.10, 0.45, 0.56),  # ship
         (0.22, 0.16, 0.27), (0.31, 0.30, 0.31),  # barriers
         (0.87, 0.4, 0.23), (0.58, 0.35, 0.28),  # reach area
         (0.25, 0.41, 0.88), (0, 0, 0)]

RAY_CAST_LASER_NUM = 24

SHIP_POLY_BP = [
    (-5, +8), (-5, -8), (0, -8),
    (+8, -6), (+8, +6), (0, +8)
    ]

SHIP_POSITION = [(-6.5, 8), (-6.5, 1.5), (6.5, 8),
                 (6.5, 14.5), (-6.5, 14.5),
                 (0, 14.5)]

element_wise_weight = 0.8
SHIP_POLY = [
    (SHIP_POLY_BP[0][0]*element_wise_weight, SHIP_POLY_BP[0][1]*element_wise_weight),
    (SHIP_POLY_BP[1][0]*element_wise_weight, SHIP_POLY_BP[1][1]*element_wise_weight),
    (SHIP_POLY_BP[2][0]*element_wise_weight, SHIP_POLY_BP[2][1]*element_wise_weight),
    (SHIP_POLY_BP[3][0]*element_wise_weight, SHIP_POLY_BP[3][1]*element_wise_weight),
    (SHIP_POLY_BP[4][0]*element_wise_weight, SHIP_POLY_BP[4][1]*element_wise_weight),
    (SHIP_POLY_BP[5][0]*element_wise_weight, SHIP_POLY_BP[5][1]*element_wise_weight)
    ]

REACH_POLY = [(-1, 1), (-1, -1), (1, -1), (1, 1)]

RECH_RECT = [
    (-0.5, +0.5), (-0.5, -0.5),
    (+0.5, -0.5), (+0.5, +0.5)
]

action = [0, 0]


class RayCastClosestCallback(b2RayCastCallback):
    """This callback finds the closest hit"""

    def __repr__(self):
        return 'Closest hit'

    def __init__(self, **kwargs):
        b2RayCastCallback.__init__(self, **kwargs)
        self.fixture = None
        self.hit = False

    def ReportFixture(self, fixture, point, normal, fraction):
        """
        Called for each fixture found in the query. You control how the ray
        proceeds by returning a float that indicates the fractional length of
        the ray. By returning 0, you set the ray length to zero. By returning
        the current fraction, you proceed to find the closest point. By
        returning 1, you continue with the original ray clipping. By returning
        -1, you will filter out the current fixture (the ray will not hit it).
        """
        self.hit = True
        self.fixture = fixture
        self.point = b2Vec2(point)
        self.normal = b2Vec2(normal)
        # NOTE: You will get this error:
        #   "TypeError: Swig director type mismatch in output value of
        #    type 'float32'"
        # without returning a value
        return fraction


class ContactDetector(b2ContactListener):
    def __init__(self, env):
        b2ContactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.ship == contact.fixtureA.body or self.env.ship == contact.fixtureB.body:
            self.env.ship.contact = True
            if self.env.reach_area == contact.fixtureA.body or self.env.reach_area == contact.fixtureB.body:
                self.env.game_over = True
            elif self.env.ground == contact.fixtureA.body or self.env.ground == contact.fixtureB.body:
                self.env.ground_contact = True

    def EndContact(self, contact):
        if self.env.ship in [contact.fixtureA.body, contact.fixtureB.body]:
            self.env.ship.contact = False


class PosIter:
    def __init__(self, x):
        self.val = x
        self.next = None


class RoutePlan(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self, barrier_num=3, seed=None, ship_pos_fixed=None, positive_heatmap=None, barrier_radius=1, test=False):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.seed_num = seed
        self.ship_pos_fixed = ship_pos_fixed
        if positive_heatmap is not None:
            self.positive_heat_map = True
        else:
            self.positive_heat_map = None
        self.hard_update = False
        self.test = test

        # 环境物理结构变量
        self.world = Box2D.b2World(gravity=(0, 0))
        self.barrier = []
        self.reef = []
        self.end = None
        self.ship = None
        self.reach_area = None
        self.ground = None

        # 障碍物数量
        self.barrier_num = barrier_num
        self.barrier_radius = barrier_radius
        # 障碍物生成边界
        self.barrier_bound_x = 0.6
        self.barrier_bound_y = 0.8
        self.dead_area_bound = 0.03
        self.ship_radius = 0.36*element_wise_weight

        # game状态记录
        self.timestep = 0
        self.game_over = None
        self.ground_contact = None
        self.dist_record = None
        self.draw_list = None
        self.dist_norm = 14.38
        self.dist_init = None
        self.iter_ship_pos = None

        # 生成环形链表
        self.loop_ship_posGenerator()

        # heatmap生成状态记录
        self.heatmap_mapping_ra = None
        self.heatmap_mapping_bl = None
        self.heat_map = None
        self.pathFinding = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        self.reset()

    def loop_ship_posGenerator(self):
        self.iter_ship_pos = cur = PosIter(SHIP_POSITION[0])
        for x in SHIP_POSITION:
            tmp = PosIter(x)
            cur.next = tmp
            cur = tmp
        cur.next = self.iter_ship_pos

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if self.ship is None:
            return
        # 清除障碍物
        if self.barrier:
            for idx, barr in enumerate(self.barrier):
                self.world.DestroyBody(barr)
        self.barrier.clear()
        if self.reef:
            for reef in self.reef:
                self.world.DestroyBody(reef)
        self.reef.clear()
        # 清除reach area
        self.world.DestroyBody(self.reach_area)
        self.reach_area = None
        # 清除船体
        self.world.DestroyBody(self.ship)
        self.ship = None

    def isValid(self, fixture_center, barrier_dict, reef_dict):
        for idx in range(self.barrier_num):
            if Distance_Cacul(barrier_dict['center_point'][idx], fixture_center) > 2.5 * barrier_dict['radius'][idx]:
                continue
            else:
                return False
        for idx_reef in range(len(reef_dict['center_point'])):
            if Distance_Cacul(reef_dict['center_point'][idx_reef], fixture_center) > 1.25:
                continue
            else:
                return False
        return True

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.ground_contact = False
        self.dist_record = None
        self.timestep = 0

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE
        """设置边界范围"""
        self.ground = self.world.CreateBody(position=(0, VIEWPORT_H/SCALE/2))
        self.ground.CreateEdgeFixture(vertices=[(-VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2),
                                                (-VIEWPORT_W/SCALE/2, -VIEWPORT_H/SCALE/2),
                                                (VIEWPORT_W/SCALE/2, -VIEWPORT_H/SCALE/2),
                                                (VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2)],
                                      friction=1.0,
                                      density=1.0)
        self.ground.CreateEdgeChain(
            [(-VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2),
             (-VIEWPORT_W/SCALE/2, -VIEWPORT_H/SCALE/2),
             (VIEWPORT_W/SCALE/2, -VIEWPORT_H/SCALE/2),
             (VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2),
             (-VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2)])

        """设置障碍物位置转移mesh"""
        # 存储生成barrier参数
        barrier_dict = {'center_point': [],
                        'radius': []}
        CHUNKS = 3
        start_center = (-(W/2 - W*(1 - self.barrier_bound_x) / 2 - W * self.barrier_bound_x / CHUNKS * 1/2),
                        (H - (1 - self.barrier_bound_y) * 1/2 * H - (H * self.barrier_bound_y) / CHUNKS * 1/2))
        shift_x, shift_y = np.linspace(start_center[0], -start_center[0], CHUNKS),\
                           np.linspace(start_center[1], (H / CHUNKS * 1/2 + H * (1 - self.barrier_bound_y) * 0.5), CHUNKS)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        if self.seed_num is not None:
            np.random.seed(self.seed_num)
        index = np.random.choice([i for i in range(CHUNKS ** 2)], self.barrier_num + 1, replace=False)
        # print(index)
        # x, y = [], []
        for idxbr in index[:-1]:
            # 控制障碍物生成位置在圈定范围之内60％部分
            random_noise_x = np.random.uniform(-W*self.barrier_bound_x*0.05, W*self.barrier_bound_x*0.05)
            random_noise_y = np.random.uniform(-H*self.barrier_bound_y*0.05, H*self.barrier_bound_y*0.05)
            # 通过index选择障碍物位置
            radius = self.barrier_radius * np.random.uniform(0.2 + 1.3**(-self.barrier_num), 0.5 + 1.08**(-self.barrier_num))
            barrier_pos = (shift_x[idxbr // CHUNKS, idxbr % CHUNKS] + random_noise_x,
                           shift_y[idxbr // CHUNKS, idxbr % CHUNKS] + random_noise_y)
            self.barrier.append(
                self.world.CreateStaticBody(shapes=b2CircleShape(
                    pos=barrier_pos,
                    radius=radius)))
            # 记录障碍物参数
            barrier_dict['center_point'].append(barrier_pos)
            barrier_dict['radius'].append(radius)
        #     x.append(shift_x[idxbr // CHUNKS, idxbr % CHUNKS])
        #     y.append(shift_y[idxbr // CHUNKS, idxbr % CHUNKS])
        # import matplotlib.pyplot as plt
        # plt.scatter(x, y)
        # plt.show()

        """暗礁生成"""
        # reef generate test
        # reef_dict = {'center_point': [],
        #              'radius': []}
        # if self.seed_num is not None:
        #     self.np_random.seed(self.seed_num)
        # while len(self.reef) < 5:
        #     reef_position = (self.np_random.uniform(-5, 2),
        #                      self.np_random.uniform(8, 13))
        #     if self.isValid(reef_position, barrier_dict, reef_dict):
        #         reef = self.world.CreateStaticBody(shapes=b2CircleShape(pos=reef_position,
        #                                            radius=0.36))
        #         reef.hide = True
        #         self.reef.append(reef)
        #         reef_dict['center_point'].append(reef_position)
        #         reef_dict['radius'].append(0.36)

        """ship生成"""
        """!!!   已验证   !!!"""
        initial_position_x, initial_position_y = None, None
        if not self.ship_pos_fixed:
            # if self.seed_num is not None:
            #     self.np_random.seed(self.seed_num)
            # initial_position_x = self.np_random.uniform(-W * (0.5 - self.dead_area_bound),
            #                                             -W * self.barrier_bound_x / 2)
            initial_position_x = self.np_random.uniform(-W * (0.5 - self.dead_area_bound),
                                                        W * self.barrier_bound_x/2)
            if initial_position_x < - 1/2 * self.barrier_bound_x * W:
                # if self.seed_num is not None:
                #     self.np_random.seed(self.seed_num)
                initial_position_y = self.np_random.uniform(H * self.dead_area_bound,
                                                            H * (1 - self.dead_area_bound))
            elif - 1/2 * self.barrier_bound_x * W < initial_position_x:
                # if self.seed_num is not None:
                #     self.np_random.seed(self.seed_num)
                initial_position_y = random.choice([self.np_random.uniform(H * self.dead_area_bound, H * (1-self.barrier_bound_y)),
                                      (self.np_random.uniform(H * self.barrier_bound_y, H * (1-self.dead_area_bound)))])
        else:
            # 判断worker是否使用循环
            random_position = self.iter_ship_pos.val
            initial_position_x, initial_position_y = random_position[0], random_position[1]
            if self.iter_ship_pos.next is None:
                self.end = True
            else:
                self.iter_ship_pos = self.iter_ship_pos.next
        """
        >>>help(Box2D.b2BodyDef)
        angularDamping: 角度阻尼
        angularVelocity: 角速度
        linearDamping：线性阻尼
        #### 增加线性阻尼可以使物体行动有摩擦
        """
        self.ship = self.world.CreateDynamicBody(
            position=(initial_position_x, initial_position_y),
            angle=0.0,
            angularDamping=20,
            linearDamping=10,
            fixedRotation=True,
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(vertices=[(x/SCALE, y/SCALE) for x, y in SHIP_POLY]),
                density=1 if self.test else 1,
                friction=1,
                categoryBits=0x0010,
                maskBits=0x001,     # collide only with ground
                restitution=0.0)    # 0.99 bouncy
                )
        self.ship.contact = False
        self.ship.color_bg = PANEL[4]
        self.ship.color_fg = PANEL[5]
        self.ship.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                                      self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)), wake=True)

        """抵达点生成"""
        # 设置抵达点位置
        reach_center_x = W/2 * 0.6
        reach_center_y = H*0.75
        # circle_shape = b2CircleShape(radius=0.85)
        circle_shape = b2PolygonShape(vertices=[(x/2, y/2) for x, y in REACH_POLY])
        self.reach_area = self.world.CreateStaticBody(position=(reach_center_x, reach_center_y),
                                                      fixtures=b2FixtureDef(
                                                          shape=circle_shape
                                                      ))
        self.reach_area.color = PANEL[4]
        self.draw_list = [self.ship] + self.barrier + [self.reach_area] + [self.ground] + self.reef

        # reward Heatmap构建
        # 使heatmap只生成一次
        # profile测试发现heatmap生成所消耗时间是Box2D环境生成所消耗时间3000倍以上
        # 减少heatmap在reset中重置次数
        if not self.hard_update:
            bound_list = self.barrier + self.reef + [self.reach_area] + [self.ground]
            heat_map_init = HeatMap(bound_list, positive_reward=self.positive_heat_map)
            self.pathFinding = heat_map_init.rewardCal4TraditionalMethod
            self.heatmap_mapping_ra = heat_map_init.ra
            self.heatmap_mapping_bl = heat_map_init.bl
            self.heat_map = heat_map_init.rewardCal
            self.heat_map += heat_map_init.ground_rewardCal_redesign
            self.heat_map += heat_map_init.reach_rewardCal
            self.heat_map = normalize(self.heat_map) - 1
            self.hard_update = True
        """seaborn heatmap"""
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # fig, axes = plt.subplots(1, 1)
        # sns.heatmap(self.heat_map.T, annot=False, ax=axes).invert_yaxis()
        # plt.show()
        """matplotlib 3d heatmap"""
        # from mpl_toolkits import mplot3d
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # x, y = np.meshgrid(np.linspace(0, 159, 160), np.linspace(0, 159, 160))
        # ax.plot_surface(x, y, self.heat_map.T, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        # plt.show()

        end_info = b2Distance(shapeA=self.ship.fixtures[0].shape,
                              idxA=0,
                              shapeB=self.reach_area.fixtures[0].shape,
                              idxB=0,
                              transformA=self.ship.transform,
                              transformB=self.reach_area.transform,
                              useRadii=True)
        self.dist_init = end_info.distance
        return self.step(self.np_random.uniform(-1, 1, size=(2,)))

    def step(self, action_sample: np.array):
        action_sample = np.clip(action_sample, -1, 1).astype('float32')

        if not self.ship:
            return

        """船体推进位置及动力大小计算"""
        force2ship = self.remap(action_sample[0], MAIN_ENGINE_POWER)
        orient2ship = self.remap(action_sample[1], MAIN_ORIENT_POWER)
        orient_position = (math.cos(orient2ship)*force2ship, math.sin(orient2ship)*force2ship)
        # 计算船体local vector相对于世界vector方向
        force2ship = self.ship.GetWorldVector(localVector=(orient_position[0], orient_position[1]))

        # 获取力量点位置
        force2position = self.ship.GetWorldPoint(localPoint=(0, 0))

        self.ship.ApplyForce(force2ship, force2position, True)
        self.world.Step(1.0 / FPS, 10, 10)

        """
        # 取余操作在对负数取余时，在Python当中,如果取余的数不能够整除，那么负数取余后的结果和相同正数取余后的结果相加等于除数。
        # 将负数角度映射到正确的范围内
        if self.ship.angle < 0:
            angle_unrotate = - ((b2_pi*2) - self.ship.angle % (b2_pi * 2))
        else:
            angle_unrotate = self.ship.angle % (b2_pi * 2)
        # 角度映射到 [-pi, pi]
        if angle_unrotate < -b2_pi:
            angle_unrotate += (b2_pi * 2)
        elif angle_unrotate > b2_pi:
            angle_unrotate -= (b2_pi * 2)

        vel_temp = self.ship.linearVelocity
        # 计算船体行进方向的单位向量相对world向量
        ship_unit_vect = self.ship.GetWorldVector(localVector=(1.0, 0.0))
        # 计算速度方向到单位向量的投影，也就是投影在船轴心x上的速度
        vel2ship_proj = b2Dot(ship_unit_vect, vel_temp)
        """

        # # 11 维传感器数据字典
        # sensor_raycast = {"points": np.zeros((RAY_CAST_LASER_NUM, 2)),
        #                   'normal': np.zeros((RAY_CAST_LASER_NUM, 2)),
        #                   'distance': np.zeros((RAY_CAST_LASER_NUM, 2))}
        # # 传感器扫描
        # length = self.ship_radius * 10      # Set up the raycast line
        # point1 = self.ship.position
        # for vect in range(RAY_CAST_LASER_NUM):
        #     ray_angle = self.ship.angle - b2_pi/2 + (b2_pi*2/RAY_CAST_LASER_NUM * vect)
        #     d = (length * math.cos(ray_angle), length * math.sin(ray_angle))
        #     point2 = point1 + d
        #
        #     # 初始化Raycast callback函数
        #     callback = RayCastClosestCallback()
        #
        #     self.world.RayCast(callback, point1, point2)
        #
        #     if callback.hit:
        #         sensor_raycast['points'][vect] = callback.point
        #         sensor_raycast['normal'][vect] = callback.normal
        #         if callback.fixture == self.reach_area.fixtures[0]:
        #             sensor_raycast['distance'][vect] = (3, Distance_Cacul(point1, callback.point) - self.ship_radius)
        #         elif callback.fixture in self.ground.fixtures:
        #             sensor_raycast['distance'][vect] = (2, Distance_Cacul(point1, callback.point) - self.ship_radius)
        #         else:
        #             sensor_raycast['distance'][vect] = (1, Distance_Cacul(point1, callback.point) - self.ship_radius)
        #     else:
        #         sensor_raycast['distance'][vect] = (0, 10*self.ship_radius)
        # sensor_raycast['distance'][..., 1] /= self.ship_radius*10

        pos = self.ship.position
        try:
            pos_mapping = heat_map_trans(pos)
        except ValueError as e:
            print('pos value error with Nan')
        vel = self.ship.linearVelocity

        # 基于polyshape的最近距离测算
        end_info = b2Distance(shapeA=self.ship.fixtures[0].shape,
                              idxA=0,
                              shapeB=self.reach_area.fixtures[0].shape,
                              idxB=0,
                              transformA=self.ship.transform,
                              transformB=self.reach_area.transform,
                              useRadii=True)
        end_ori = Orient_Cacul(end_info[1], end_info[0])
        # print(end_info[2])

        # ship速度计算
        vel_scalar = Distance_Cacul(vel, b2Vec2(0, 0))

        """        
        # 状态值归一化
        state = [
            (pos.x - self.reach_area.position.x)/8,
            (pos.y - self.reach_area.position.y)/16,
            vel_scalar,
            end_ori/b2_pi,
            end_info.distance/self.dist_norm,
            [sensor_info for sensor_info in sensor_raycast['distance'].reshape(-1)]
        ]
        assert len(state) == 6
        """
        # state = [
        #     end_info.distance,
        #     end_ori/b2_pi,
        #     [sensor_info for sensor_info in sensor_raycast['distance'].reshape(-1)]
        # ]
        # assert len(state) == 3

        # state = [
        #     (pos.x - self.reach_area.position.x) / 8,
        #     (pos.y - self.reach_area.position.y) / 16,
        #     end_info.distance,
        #     end_ori / b2_pi]
        # assert len(state) == 4

        state = [
            end_info.distance,
            end_ori / b2_pi]
        assert len(state) == 2

        """Reward 计算"""
        done = False

        # ship投影方向速度reward计算
        if vel_scalar > 5:
            reward_vel = -0.5
        else:
            reward_vel = 0

        if self.dist_record is not None and self.dist_record < end_info.distance:
            # reward_dist = -end_info.distance / self.dist_init
            reward_dist = -1
        else:
            # reward_dist = 1 - end_info.distance / self.dist_init
            reward_dist = 1
            self.dist_record = end_info.distance

        reward = self.heat_map[pos_mapping[0], pos_mapping[1]] + reward_dist
        # print(f'reward_heat:{reward_shaping:.3f}, reward_dist: {reward_dist:.3f}')

        # 定义成功终止状态
        if self.ship.contact:
            if self.game_over:
                reward = 10
                done = True
            elif self.ground_contact:
                reward = -10
                done = True
            else:
                reward = -5
                done = True
        self.timestep += 1

        '''失败终止状态定义在训练迭代主函数中，由主函数给出失败终止状态惩罚reward'''
        return np.hstack(state), reward, done, {}

    def render(self, mode='human', hide=True):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(-VIEWPORT_W/SCALE/2, VIEWPORT_W/SCALE/2, 0, VIEWPORT_H/SCALE)

        for obj in self.draw_list:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is b2CircleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    if hasattr(obj, 'hide'):
                        if not hide:
                            self.viewer.draw_circle(f.shape.radius,
                                                    20,
                                                    color=PANEL[3],
                                                    filled=False,
                                                    linewidth=2).add_attr(t)
                    else:
                        self.viewer.draw_circle(f.shape.radius, 20, color=PANEL[2]).add_attr(t)
                        self.viewer.draw_circle(f.shape.radius, 20, color=PANEL[3], filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    if hasattr(obj, 'color_bg'):
                        self.viewer.draw_polygon(path, color=obj.color_bg)
                        path.append(path[0])
                        self.viewer.draw_polyline(path, color=obj.color_fg, linewidth=2)
                    elif hasattr(obj, 'color'):
                        self.viewer.draw_polygon(path, color=PANEL[4])
                        path.append(path[0])
                        self.viewer.draw_polyline(path, color=PANEL[5], linewidth=2)
                    else:
                        self.viewer.draw_polygon(path, color=PANEL[4])
                        self.viewer.draw_polyline(path, color=PANEL[7], linewidth=10)
        # return self.viewer.render(return_rgb_array=mode == 'rgb_array')
        return self.viewer.render(return_rgb_array=True)

    @staticmethod
    def remap(action, remap_range):
        # assert isinstance(action, np.ndarray)
        return (remap_range * 2) / (1.0 * 2) * (action + 1) - remap_range

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def Distance_Cacul(pointA, pointB):
    pointA = np.array(pointA, dtype='float')
    pointB = np.array(pointB, dtype='float')
    cx = pointA[0] - pointB[0]
    cy = pointA[1] - pointB[1]
    return math.sqrt(cx * cx + cy * cy)


def Orient_Cacul(pointA, pointB):
    cx = pointA[0] - pointB[0]
    cy = pointA[1] - pointB[1]
    return math.atan2(cy, cx)


def manual_control(key):
    global action
    if key.event_type == 'down' and key.name == 'a':
        if key.event_type == 'down' and key.name == "w":
            action[1] = 1
        elif key.event_type == 'down' and key.name == 's':
            action[1] = -1
        action[0] = -1
    if key.event_type == 'down' and key.name == 'd':
        if key.event_type == 'down' and key.name == "w":
            action[1] = 1
        elif key.event_type == 'down' and key.name == 's':
            action[1] = -1
        action[0] = 1


def demo_route_plan(env, seed=None, render=False):
    # global action
    env.seed(seed)

    # ---------------------------------------------------------------- #
    env.ship.position = b2Vec2(1, 6.5)
    env.ship.angle = -0.16
    # ---------------------------------------------------------------- #

    total_reward = 0
    steps = 0
    # keyboard.hook(manual_control)
    while True:

        if not steps % 5:
            action = np.zeros((2, ))
            s, r, done, info = env.step(action)

            total_reward += r

        if render:
            still_open = env.render()
            if still_open is False:
                break

        # if steps % 20 == 0 or done:
        #     print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
        #     print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done and env.game_over:
            break
    env.close()
    return total_reward


def demo_TraditionalPathPlanning(env, seed=None, render=False):
    env.seed(seed)
    from pathfinding.core.diagonal_movement import DiagonalMovement
    from pathfinding.core.grid import Grid
    from pathfinding.finder.a_star import AStarFinder
    from pathfinding.finder.dijkstra import DijkstraFinder
    from pathfinding.finder.ida_star import IDAStarFinder
    from PIL import Image, ImageDraw

    grid = Grid(matrix=env.pathFinding)
    ship_position = heat_map_trans(env.ship.position)
    start_point = grid.node(ship_position[0], ship_position[1])
    end_point = grid.node(env.heatmap_mapping_ra['position'][0], env.heatmap_mapping_ra['position'][1])

    print(f'current height: {grid.height}, current width: {grid.width}')
    finder = DijkstraFinder(diagonal_movement=DiagonalMovement.always)
    start_time = time.time()
    path, runs = finder.find_path(start_point, end_point, grid)
    end_time = time.time() - start_time
    print(end_time)
    print('operations:', runs, 'path length:', len(path))
    print(grid.grid_str(path=path, start=start_point, end=end_point))
    print(path)


if __name__ == '__main__':
    # demo_route_plan(RoutePlan(seed=42), render=True)
    demo_TraditionalPathPlanning(RoutePlan(barrier_num=3, seed=42))
