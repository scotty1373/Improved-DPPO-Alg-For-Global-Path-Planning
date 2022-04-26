# -*-  coding=utf-8 -*-
# @Time : 2022/4/23 16:47
# @Author : Scotty1373
# @File : temp.py
# @Software : PyCharm
ori_common = self.layer_ori(state)
ori_mean = self.ori_fc1(ori_common)
ori_mean = self.ori_fc1act(ori_mean)
ori_mean = self.ori_fc2(ori_mean)
ori_mean = self.ori_fc2act(ori_mean)
ori_mean = self.ori_fc3(ori_mean)
ori_mean = self.ori_fc3act(ori_mean)

ori_std = self.ori_std_fc1(ori_common)
ori_std = self.ori_std_fc1act(ori_std)
ori_std = self.ori_std_fc2(ori_std)
ori_std = self.ori_std_fc2act(ori_std)