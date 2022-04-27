# -*-  coding=utf-8 -*-
# @Time : 2022/4/10 10:46
# @Author : Scotty1373
# @File : sea_env.py
# @Software : PyCharm
import math
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

SCALE = 30
FPS = 60

VIEWPORT_W = 480
VIEWPORT_H = 480

INITIAL_RANDOM = 20
MAIN_ENGINE_POWER = 50
MAIN_ORIENT_POWER = 10
SIDE_ENGINE_POWER = 5

#           Background           PolyLine
PANEL = [(0.19, 0.72, 0.87), (0.10, 0.45, 0.56),  # shipddddd
         (0.22, 0.16, 0.27), (0.31, 0.30, 0.31),  # barriers
         (0.87, 0.4, 0.23), (0.58, 0.35, 0.28),  # reach area
         (0.25, 0.41, 0.88)]

RAY_CAST_LASER_NUM = 26

SHIP_POLY = [
    (-5, +8), (-5, -8), (0, -8),
    (+8, -6), (+8, +6), (0, +8)
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

    def EndContact(self, contact):
        if self.env.ship in [contact.fixtureA.body, contact.fixtureB.body]:
            self.env.ship.contact = False


class RoutePlan(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self, barrier_num=3, seed=None):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None
        self.seed_num = seed

        self.world = Box2D.b2World(gravity=(0, 0))
        self.barrier = []
        self.ship = None
        self.reach_area = None
        self.ground = None

        # 障碍物数量
        self.barrier_num = barrier_num
        # 障碍物生成边界
        self.barrier_bound = 0.6
        self.dead_area_bound = 0.03
        self.ship_radius = 0.36

        # game状态记录
        self.game_over = None
        self.prev_reward = None
        self.draw_list = None
        self.dist_norm = 14.38

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        self.reset()

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
        # 清除reach area
        self.world.DestroyBody(self.reach_area)
        self.reach_area = None
        # 清除船体
        self.world.DestroyBody(self.ship)
        self.ship = None

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False

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
        CHUNKS = 3
        start_center = (-(W/2 - W*0.2 - W * self.barrier_bound / CHUNKS * 1/2),
                        (H - (1 - self.barrier_bound) * 1/2 * H - (H * self.barrier_bound) / CHUNKS * 1/2))
        shift_x, shift_y = np.linspace(start_center[0], -start_center[0], CHUNKS),\
                           np.linspace(start_center[1], (H / CHUNKS * 1/2), CHUNKS)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        if self.seed_num is not None:
            np.random.seed(self.seed_num)
        index = np.random.choice([i for i in range(CHUNKS ** 2)], self.barrier_num, replace=False)
        # print(index)
        # x, y = [], []
        for idxbr in index:
            # 控制障碍物生成位置在圈定范围之内60％部分
            if self.seed_num is not None:
                self.np_random.seed(self.seed_num)
            random_noise_x = self.np_random.uniform(-W*self.barrier_bound*0.05, W*self.barrier_bound*0.05)
            if self.seed_num is not None:
                self.np_random.seed(self.seed_num)
            random_noise_y = self.np_random.uniform(-H*self.barrier_bound*0.05, H*self.barrier_bound*0.05)
            # 通过index选择障碍物位置
            self.barrier.append(
                self.world.CreateStaticBody(shapes=b2CircleShape(
                    pos=(shift_x[idxbr // CHUNKS, idxbr % CHUNKS]+random_noise_x,
                         shift_y[idxbr // CHUNKS, idxbr % CHUNKS]+random_noise_y),
                    radius=1.2)))
        #     x.append(shift_x[idxbr // CHUNKS, idxbr % CHUNKS])
        #     y.append(shift_y[idxbr // CHUNKS, idxbr % CHUNKS])
        # import matplotlib.pyplot as plt
        # plt.scatter(x, y)
        # plt.show()

        """ship生成"""
        """!!!   已验证   !!!"""
        if self.seed_num is not None:
            self.np_random.seed(self.seed_num)
        initial_position_x = self.np_random.uniform(-W * (0.5 - self.dead_area_bound),
                                                    -W * self.barrier_bound / 2)
        if self.seed_num is not None:
            self.np_random.seed(self.seed_num)
        initial_position_y = self.np_random.uniform(H * self.dead_area_bound,
                                                    H * (1 - self.dead_area_bound))
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
            linearDamping=2,
            fixtures=b2FixtureDef(
                shape=b2PolygonShape(vertices=[(x/SCALE, y/SCALE) for x, y in SHIP_POLY]),
                density=10.0,
                friction=12,
                categoryBits=0x0010,
                maskBits=0x001,     # collide only with ground
                restitution=0.0)    # 0.99 bouncy
                )
        self.ship.contact = False
        self.ship.color_bg = PANEL[0]
        self.ship.color_fg = PANEL[1]
        self.ship.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
                                      self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)), wake=True)

        """抵达点生成"""
        # 设置抵达点位置
        """        
        reach_center_x = W/2 - W*0.05
        reach_center_y = H*0.5
        # 设置抵达点长宽，建立形状
        rect = b2PolygonShape()
        rect.SetAsBox(W*0.05, H*0.4)
        # 生成reach_area位置
        self.reach_area = self.world.CreateStaticBody(position=(reach_center_x, reach_center_y),
                                                      fixtures=b2FixtureDef(
                                                          shape=rect))
        self.reach_area.color = PANEL[4]
        """

        reach_center_x = W/2
        reach_center_y = H*0.5
        circle_shape = b2CircleShape(radius=1.2)
        self.reach_area = self.world.CreateStaticBody(position=(reach_center_x, reach_center_y),
                                                      fixtures=b2FixtureDef(
                                                          shape=circle_shape
                                                      ))
        self.reach_area.color = PANEL[4]
        self.draw_list = [self.ship] + self.barrier + [self.reach_area] + [self.ground]

        return self.step(np.array([0, 0]))

    def step(self, action: np.array):
        action = np.clip(action, -1, 1).astype('float32')

        if not self.ship:
            return

        """船体转向计算"""
        orient2ship_ori = self.remap(action[1], MAIN_ORIENT_POWER)
        # 计算船体local vector相对于世界vector方向
        # orient2ship = self.ship.GetWorldVector(localVector=(0.0, orient2ship_ori))
        # orient2position = self.ship.GetWorldPoint(localPoint=(0, 1))
        self.ship.ApplyTorque(orient2ship_ori, True)

        """船体推进位置及动力大小计算"""
        force2ship = self.remap(action[0], MAIN_ENGINE_POWER)
        # 计算船体local vector相对于世界vector方向
        force2ship = self.ship.GetWorldVector(localVector=(force2ship, 0.0))
        # 获取力量点位置
        force2position = self.ship.GetWorldPoint(localPoint=(0, 0))

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

        '''***** 当船体速度减至0时，不进行后退操作 *****'''
        # if vel2ship_proj < 0:
        #     if action[0] > 0:
        #         self.ship.ApplyForce(force2ship, force2position, True)
        #
        # else:
        #     self.ship.ApplyForce(force2ship, force2position, True)
        self.ship.ApplyForce(force2ship, force2position, True)

        self.world.Step(1.0 / FPS, 10, 10)

        # 11 维传感器数据字典
        sensor_raycast = {"points": np.zeros((RAY_CAST_LASER_NUM, 2)),
                          'normal': np.zeros((RAY_CAST_LASER_NUM, 2)),
                          'distance': np.zeros((RAY_CAST_LASER_NUM,))}
        """传感器扫描"""
        for vect in range(RAY_CAST_LASER_NUM):
            ray_angle = self.ship.angle - b2_pi/2 + (b2_pi*2/RAY_CAST_LASER_NUM * vect)
            # Set up the raycast line
            length = self.ship_radius*20
            point1 = self.ship.position
            d = (length * math.cos(ray_angle), length * math.sin(ray_angle))
            point2 = point1 + d

            # 初始化Raycast callback函数
            callback = RayCastClosestCallback()

            self.world.RayCast(callback, point1, point2)

            if callback.hit:
                sensor_raycast['points'][vect] = callback.point
                sensor_raycast['normal'][vect] = callback.normal
                sensor_raycast['distance'][vect] = Distance_Cacul(point1, callback.point) - self.ship_radius
                if callback.fixture == self.reach_area.fixtures[0]:
                    sensor_raycast['distance'][vect] = 25*self.ship_radius - self.ship_radius
            else:
                sensor_raycast['distance'][vect] = 20*self.ship_radius - self.ship_radius
        # print(sensor_raycast['distance'])

        pos = self.ship.position
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
        # ship角速度
        vel_ang = self.ship.angularVelocity

        # 状态值归一化
        # state = [
        #     (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
        #     (pos.y - VIEWPORT_H/SCALE/2) / (VIEWPORT_H/SCALE/2),
        #     vel_scalar/FPS,
        #     angle_unrotate/b2_pi,
        #     end_info.distance/self.dist_norm,
        #     [sensor_info for sensor_info in sensor_raycast['distance']]
        # ]
        state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - VIEWPORT_H/SCALE) / (VIEWPORT_H/SCALE),
            vel.x/FPS,
            vel.y/FPS,
            angle_unrotate/b2_pi,
            end_ori,
            (self.reach_area.position.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (self.reach_area.position.y - VIEWPORT_H/SCALE) / (VIEWPORT_H/SCALE),
            end_info.distance/self.dist_norm,
            [sensor_info for sensor_info in sensor_raycast['distance']]
        ]
        assert len(state) == 10

        """Reward 计算"""
        done = False

        # ship当前位置与reach area之间距离的reward计算
        reward_dist = -end_info.distance / self.dist_norm * 5
        # ship与障碍物reward计算
        sensor_rd_record = np.zeros(RAY_CAST_LASER_NUM)
        for idx, tp_dist in enumerate(sensor_raycast['distance']):
            if tp_dist - self.ship_radius <= 0:
                sensor_rd_record[idx] = -5
            if 0 < tp_dist - self.ship_radius < 1.5*self.ship_radius:
                sensor_rd_record[idx] = -3
            if self.ship_radius*1.5 < tp_dist - self.ship_radius < self.ship_radius*3:
                sensor_rd_record[idx] = -1.5
            if self.ship_radius*5 < tp_dist - self.ship_radius:
                sensor_rd_record[idx] = 0
        # 应用角度权重
        # 前部权重累加
        sensor_rd_record[6] *= 0.2
        sensor_rd_record[[5, 7]] *= 0.15
        sensor_rd_record[[4, 8]] *= 0.1
        # 后部权重累加
        sensor_rd_record[[17, 21]] *= 0.1
        sensor_rd_record[[18, 20]] *= 0.15
        sensor_rd_record[19] *= 0.2

        sensor_rd_record[0:4] *= 0.05
        sensor_rd_record[9:17] *= 0.05
        sensor_rd_record[22:] *= 0.05
        reward_coll = sensor_rd_record.sum()

        # ship角速度reward计算
        reward_ang_vel = -abs(vel_ang)

        # ship速度reward计算
        if vel_scalar > 3:
            reward_vel = -3
        else:
            reward_vel = 0

        reward_unrotate = -abs(angle_unrotate / b2_pi)

        reward = reward_coll + reward_dist + reward_ang_vel
        # print(f'reward_coll:{reward_coll}, reward_dist:{reward_dist}, reward_vel:{reward_vel}')

        # 定义成功终止状态
        if self.ship.contact:
            if self.game_over:
                reward = 100
            else:
                reward = -20
            done = True

        '''失败终止状态定义在训练迭代主函数中，由主函数给出失败终止状态惩罚reward'''
        return np.hstack(state), reward, done, {}

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(-VIEWPORT_W/SCALE/2, VIEWPORT_W/SCALE/2, 0, VIEWPORT_H/SCALE)

        for obj in self.draw_list:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is b2CircleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    # reach area区域渲染
                    if hasattr(obj, 'color'):
                        self.viewer.draw_circle(f.shape.radius, 20, color=PANEL[4]).add_attr(t)
                        self.viewer.draw_circle(f.shape.radius, 20, color=PANEL[5], filled=False, linewidth=2).add_attr(t)
                    else:
                        self.viewer.draw_circle(f.shape.radius, 20, color=PANEL[2]).add_attr(t)
                        self.viewer.draw_circle(f.shape.radius, 20, color=PANEL[3], filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    if hasattr(obj, 'color_bg'):
                        self.viewer.draw_polygon(path, color=obj.color_bg)
                        path.append(path[0])
                        self.viewer.draw_polyline(path, color=obj.color_fg, linewidth=2)
                    else:
                        self.viewer.draw_polygon(path, color=PANEL[4])
                        self.viewer.draw_polyline(path, color=PANEL[5], linewidth=2)
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
    cx = pointA[0] - pointB[0]
    cy = pointA[1] - pointB[1]
    return math.sqrt(cx * cx + cy * cy)


def Orient_Cacul(pointA, pointB):
    cx = pointA[0] - pointB[0]
    cy = pointA[1] - pointB[1]
    return math.atan2(cy, cx)


def manual_control(key):
    global action
    if key.event_type == "down" and key.name == 'w':
        action[1] = 1
    if key.event_type == 'down' and key.name == 's':
        action[1] = -1
    if key.event_type == 'down' and key.name == 'a':
        action[0] = -1
    if key.event_type == 'down' and key.name == 'd':
        action[0] = 1


def demo_route_plan(env, seed=None, render=False):
    global action
    env.seed(seed)
    total_reward = 0
    steps = 0
    keyboard.hook(manual_control)
    while True:

        if not steps % 5:
            s, r, done, info = env.step(action)
            action = [0, 0]
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


if __name__ == '__main__':
    demo_route_plan(RoutePlan(), render=True)
