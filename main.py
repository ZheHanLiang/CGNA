##########################################################################
# @File: main.py
# @Author: Zhehan Liang
# @Date: 6/8/2020
# @Intro: 训练的主函数
##########################################################################

import time
import argparse
import torch
import networkx as nx
import numpy as np
from utils import read_graph, adjacency_matrix_normalize, get_node_num, get_two_graph_scores, load_embeddings
from initialize import initialize_seed, initialize_feature
from model import build_model, save_best_model, reload_best_model

# ## t-sne
# from tsne import tsne
# import matplotlib.pyplot as plt

## 限制显卡
import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'

## 随机数种子
np.random.seed(1)

## 参数设置
parser = argparse.ArgumentParser(description='CNNA') # 实例化ArgumenParser
parser.add_argument("--seed", type=int, default=1, help="Initialization seed(<=0 to disable)") # 使用add_argument函数添加参数
parser.add_argument("--cuda", type=bool, default=True, help="Run on GPU")
parser.add_argument("--profile_feature", type=bool, default=False, help="Using initial profile feature")
parser.add_argument("--data_path", type=str, default="./data/graph_edge/", help="Path of edge file)")
parser.add_argument("--best_model_path", type=str, default="./log/", help="Path of best model)")
parser.add_argument("--dataset", "-d", type=str, default="lastfm-lastfm", help="Names of datasets")
parser.add_argument("--model", type=str, default="CNEN", help="Training model, GCN or CNEN")
parser.add_argument("--init_dim", type=int, default=64, help="Initial feature dimension")
parser.add_argument("--emb_dim", type=int, default=32, help="Embedding dimension")
parser.add_argument("--iteration_num", type=int, default=400, help="Number of training iterations in each epoch")
parser.add_argument("--epoch_num", type=int, default=100, help="Number of training epoch")
parser.add_argument("--node_num", "-n", type=int, default=10000, help="Number of graph nodes")
parser.add_argument("--radius", "-r", type=float, default=3.0, help="Radius of Adaptive Feature Norm")
parser.add_argument("--stop_epoch_num", type=int, default=10, help="Early stop if precision goes down")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate of model")
parser.add_argument("--metric_method", type=str, default="Euclid", help="Method of matrix metric: Euclid or cosine")
parser.add_argument("--precision_1_max", type=float, default=0.0, help="Max precision of precision_1")
parser.add_argument("--precision_5_max", type=float, default=0.0, help="Max precision of precision_5")
parser.add_argument("--precision_10_max", type=float, default=0.0, help="Max precision of precision_10")
parser.add_argument("--greatest_epoch", type=int, default=0, help="Greatest epoch")
## initial profile feature
parser.add_argument("--emb_dir", type=str, default="./data/profile_feature/")
## tsne
parser.add_argument("--show_node_num", type=int, default=10)

params = parser.parse_args() # 进行参数解析

## 初始化随机数种子
initialize_seed(params)
## 获取节点数，真实数据集是固定的，合成数据集要按照生成时的参数确定
if params.dataset == 'MC3-MC3': params.node_num = 5507
# # 跨网络情况
# elif params.dataset == 'flickr-lastfm': params.node_num = 641
# elif params.dataset == 'flickr-myspace': params.node_num = 509
# elif params.dataset == 'lastfm-myspace': params.node_num = 1906
else:
    datasets = params.dataset.split('-')
    assert len(datasets)==2, "Please input datasets pair!" # 校验数据集输入的是否是数据集对
    assert datasets[0]==datasets[1], "Unkown datasets pair!" # 校验数据集输入是否正确
    params.node_num = get_node_num(params)

## 获取嵌入矩阵
if params.profile_feature:
    src_emb = load_embeddings(params, True)
    tgt_emb = load_embeddings(params, False)
else:
    src_emb, tgt_emb = initialize_feature(params)

# ## 读取图
G_source_original, G_target_original = read_graph(params)
## 将原始图赋值给G_source和G_target，作为当前图
G_source = G_source_original
G_target = G_target_original
G_source_edge_num = nx.number_of_edges(G_source)
G_target_edge_num = nx.number_of_edges(G_target)
print("=====> number of source grpah edge: %d, number of target grpah edge: %d"%(G_source_edge_num, G_target_edge_num))
A_source = nx.adjacency_matrix(G_source)
A_target = nx.adjacency_matrix(G_target)
## 读取当前图的邻接矩阵，并进行标准化，方便进行图卷积
A_source_norm, A_target_norm = adjacency_matrix_normalize(params, G_source, G_target)
## 构建网络
model = build_model(params)
## 选择adam优化器
gd = torch.optim.Adam(model.parameters(), lr=params.lr)

for epoch in range(params.epoch_num):
    ## early stop
    if epoch>params.greatest_epoch+params.stop_epoch_num: break

    ## 训练模型参数
    time_head = time.time() # 记录开始时间
    for iteration in range(params.iteration_num):
        y_s, y_t,_ = model(A_source_norm, src_emb, A_target_norm, tgt_emb)
        loss = (y_s.norm(p=2, dim=1).mean() - params.radius) ** 2 + (y_t.norm(p=2, dim=1).mean() - params.radius) ** 2
        gd.zero_grad()
        loss.backward()
        gd.step()
        if iteration%100==0:
            print(loss.data.cpu().numpy())
    time_tail = time.time() # 记录完成时间
    print("=====> epoch %d GCN training has finished!\tTime: %.3f"%(epoch+1, time_tail-time_head))
    print("=====> epoch %d GCN training loss: %f"%(epoch+1, loss.detach().cpu().numpy()))
    ## 相似度计算与最近点匹配
    y_s, y_t,_ = model(A_source_norm, src_emb, A_target_norm, tgt_emb)
    match_pairs = get_two_graph_scores(y_s.detach().cpu().numpy(), y_t.detach().cpu().numpy(), operation='match', method=params.metric_method)
    acc_1, acc_5, acc_10 = get_two_graph_scores(y_s.detach().cpu().numpy(), y_t.detach().cpu().numpy(), operation='evaluate', method=params.metric_method)
    if acc_1>params.precision_1_max:
        params.precision_1_max = acc_1
        # 以acc_1为标准记录最优模型
        save_best_model(params, model)
        params.greatest_epoch = epoch
    if acc_5>params.precision_5_max:
        params.precision_5_max = acc_5
    if acc_10>params.precision_10_max:
        params.precision_10_max = acc_10
        # # 以acc_10为标准记录最优模型
        # save_best_model(params, model)
        # params.greatest_epoch = epoch
    print("=====> matching accuracy: acc_1: %.4f, acc_5: %.4f, acc_10: %.4f"%(acc_1, acc_5, acc_10))

    print("=====> epoch %d have finished!"%(epoch+1))

## 重新载入最佳模型
model = reload_best_model(params)
y_s, y_t, S = model(A_source_norm, src_emb, A_target_norm, tgt_emb)
acc_1, acc_5, acc_10 = get_two_graph_scores(y_s.detach().cpu().numpy(), y_t.detach().cpu().numpy(), operation='evaluate', method=params.metric_method)
print("=====> Epoch %d best matching accuracy: acc_1: %.4f, acc_5: %.4f, acc_10: %.4f"%(params.greatest_epoch, acc_1, acc_5, acc_10))

################################################################################################################
################################################### t-sne ######################################################
################################################################################################################
# totle = np.r_[y_s.detach().cpu().numpy(), y_t.detach().cpu().numpy()]
# X_tsne_s = tsne(totle, 2, 50, 20.0)
# match_Pairs = []
# for n in range(params.show_node_num): match_Pairs.append(n)
# # X_tsne_s = tsne(y_s.detach().cpu().numpy(), 2, 50, 20.0)
# # X_tsne_t = tsne(y_t.detach().cpu().numpy(), 2, 50, 20.0)
# # plt.figure(figsize=(60, 60))
# # plt.scatter(X_tsne_s[0:100, 0], X_tsne_s[0:100, 1], c='#00CED1')
# # plt.scatter(X_tsne_t[0:100, 0], X_tsne_t[0:100, 1], c='#DC143C')
# plt.scatter(X_tsne_s[0:params.show_node_num, 0], X_tsne_s[0:params.show_node_num, 1], c=match_Pairs)
# plt.scatter(X_tsne_s[params.node_num:params.node_num+params.show_node_num, 0], X_tsne_s[params.node_num:params.node_num+params.show_node_num, 1], c=match_Pairs)
# print("A_x", X_tsne_s[0:params.show_node_num, 0])
# print("B_y", X_tsne_s[params.node_num:params.node_num+params.show_node_num, 1])
# print("A_y", X_tsne_s[0:params.show_node_num, 1])
# print("B_x", X_tsne_s[params.node_num:params.node_num+params.show_node_num, 0])
# plt.savefig('tsne_flickr_10_ites_GCN.pdf', dpi=600) # 保存
# plt.show()
