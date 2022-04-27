# -*-  coding=utf-8 -*-
# @Time : 2022/4/27 10:07
# @Author : Scotty1373
# @File : pixel_based.py
# @Software : PyCharm
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ActorModel(nn.Module):
    def __init__(self):
        super(ActorModel, self).__init__()