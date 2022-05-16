# -*-  coding=utf-8 -*-
# @Time : 2022/5/16 9:57
# @Author : Scotty1373
# @File : utils.py
# @Software : PyCharm
import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2gray


# 数据帧叠加
def state_frame_overlay(new_state, old_state, frame_num):
    new_frame_overlay = np.concatenate((new_state.reshape(1, -1),
                                        old_state.reshape(frame_num, -1)[:(frame_num - 1), ...]),
                                       axis=0).reshape(1, -1)
    return new_frame_overlay


# 基于图像的数据帧叠加
def pixel_based(new_state, old_state, frame_num):
    new_frame_overlay = np.concatenate((new_state,
                                       old_state[:, :(frame_num - 1), ...]),
                                       axis=1)
    return new_frame_overlay


def img_proc(img, resize=(80, 80)):
    img = Image.fromarray(img.astype(np.uint8))
    img = np.array(img.resize(resize, resample=Image.BILINEAR))
    # img = img_ori.resize(resize, resample=Image.NEAREST)
    # img.show()
    # img = img_ori.resize(resize, resample=Image.BILINEAR)
    # img.show()
    # img = img_ori.resize(resize, Image.BICUBIC)
    # img.show()
    # img = img_ori.resize(resize, Image.ANTIALIAS)
    # img.show()
    img = rgb2gray(img).reshape(1, 1, 80, 80)
    return img.copy()



