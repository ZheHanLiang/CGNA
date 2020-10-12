##########################################################################
# @File: initialize.py
# @Author: Zhehan Liang
# @Date: 6/10/2020
# @Intro: 初始化函数
##########################################################################

import numpy as np
import torch
import time

def initialize_seed(params):
    """
    初始化随机数种子
    """
    time_head = time.time() # 记录开始时间
    if params.seed>0: # 当设置了正的seed值时产生numpy和torch的随机数种子
        np.random.seed(params.seed) # numpy部分的随机数种子
        torch.manual_seed(params.seed) # torch中CPU部分的随机数种子
        if params.cuda: # 当使用cuda时
            torch.cuda.manual_seed(params.seed) # torch中GPU部分的随机数种子
    time_tail = time.time() # 记录完成时间
    print("=====> Seeds have been initialized!\tTime: %.3f"%(time_tail-time_head))

def initialize_feature(params):
    """
    初始化随机节点特征
    """
    # ## 加噪声的初始化嵌入
    # feature_s = (torch.ones(params.node_num, params.init_dim) + torch.randn(params.node_num, params.init_dim) / 10) * params.radius
    # feature_t = (torch.ones(params.node_num, params.init_dim) + torch.randn(params.node_num, params.init_dim) / 10) * params.radius
    ## 不加噪声的初始化嵌入
    feature_s = (torch.ones(params.node_num, params.init_dim)) * params.radius
    feature_t = (torch.ones(params.node_num, params.init_dim)) * params.radius
    if params.cuda:
        return feature_s.cuda(), feature_t.cuda()
    else:
        return feature_s, feature_t
