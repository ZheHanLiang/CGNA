##########################################################################
# @File: initial_data_processing_cross.py
# @Author: Zhehan Liang
# @Date: 8/18/2020
# @Intro: 对原始数据进行调整，对节点的数量、删除边节点的最小度数、数据格式等
# 进行处理，并分别删去一定数量不重复的边，最后保存为源域和目标域两个新图，保存
# 格式为.txt
# @Data source: https://www.aminer.cn/cosnet
##########################################################################

import numpy as np
import os
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Initial data processing of cross case.")
    parser.add_argument('--dataset', '-d', type=str, default='flickr-lastfm', help='dataset pair.')
    return parser.parse_args()

def data_processing(args):
    ## 参数设置
    # data_name = ["lastfm", "flickr", "myspace"] # 数据集名称列表(list)
    # dataset = data_name[args.dataset]
    datasets = args.dataset.split('-')
    dataset_s = datasets[0]
    dataset_t = datasets[1]
    input_dir = "./data/graph_data/" # 原始数据文件路径
    map_dir = "./data/graph_map/" # 映射文件路径
    output_dir = "./data/graph_edge/" # 新数据文件存储路径

    ## 路径设置
    input_data_edge_dir_s = input_dir + "{0}/{0}.edges".format(dataset_s) # 源域图边路径
    input_data_edge_dir_t = input_dir + "{0}/{0}.edges".format(dataset_t) # 目标域图边路径
    input_data_node_dir_s = input_dir + "{0}/{0}.nodes".format(dataset_s) # 源域图节点路径
    input_data_node_dir_t = input_dir + "{0}/{0}.nodes".format(dataset_t) # 目标域图节点路径
    map_file = map_dir + "{0}.map.raw".format(args.dataset) # 映射文件路径
    # output_new_data_dir = output_dir + "{0}-{0}_{1}_new.edges".format(dataset, total_num) # 保留目标数量节点文件路径
    output_source_dir = output_dir + "{0}_source_edges.txt".format(args.dataset) # 源域图保存路径
    output_target_dir = output_dir + "{0}_target_edges.txt".format(args.dataset) # 目标域图保存路径

    ## 读取映射文件，按顺序将每行的id存储为两个list，id在list中的索引就是对应在新图中的索引
    id_list_s = []
    id_list_t = []
    map_num = 0
    f = open(map_file, 'r', encoding='utf-8')
    for line in f.readlines():
        ids = line.strip().split(' ')
        assert len(ids)==2, "ids's length error!"
        map_dic = {ids[0]: ids[1]}
        id_list_s.append(ids[0])
        id_list_t.append(ids[1])
        map_num += 1
    f.close()

    ## 根据.nodes文件确定各个id在原始图中的索引
    id_index_s = {}
    id_index_t = {}
    f = open(input_data_node_dir_s, 'r', encoding='utf-8')
    flag = 0
    for line in f.readlines():
        flag += 1
        assert len(line.strip().split('\t'))<3, "content length error!{0}".format(flag)
        if len(line.strip().split('\t'))==1: print(flag); continue
        index = line.strip().split('\t')[0]
        id_str = line.strip().split('\t')[1]
        if id_str in id_list_s:
            id_index_s[index] = id_str
    f.close()
    flag = 0
    f = open(input_data_node_dir_t, 'r', encoding='utf-8')
    for line in f.readlines():
        flag += 1
        assert len(line.strip().split('\t'))<3, "content length error!{0}".format(flag)
        if len(line.strip().split('\t'))==1: print(flag); continue
        index = line.strip().split('\t')[0]
        id_str = line.strip().split('\t')[1]
        if id_str in id_list_t:
            id_index_t[index] = id_str
    f.close()

    ## 遍历.edges文件，如果某一行的两个边索引都是id_index字典中的关键词的话，就根据对应的值（id名）和id_list确定它们在新图中的索引，记录构成的边
    edges_s = ""
    edges_t = ""
    f = open(input_data_edge_dir_s, 'r', encoding='utf-8')
    for line in f.readlines():
        index_1 = line.strip().split(' ')[0]
        index_2 = line.strip().split(' ')[1]
        if index_1 in id_index_s and index_2 in id_index_s:
            id_1 = id_index_s[index_1]
            id_2 = id_index_s[index_2]
            index_1 = id_list_s.index(id_1)
            index_2 = id_list_s.index(id_2)
            edges_s += "{0} {1} \n".format(index_1, index_2)
    f.close()
    f = open(input_data_edge_dir_t, 'r', encoding='utf-8')
    for line in f.readlines():
        index_1 = line.strip().split(' ')[0]
        index_2 = line.strip().split(' ')[1]
        if index_1==index_2: continue
        if index_1 in id_index_t and index_2 in id_index_t:
            id_1 = id_index_t[index_1]
            id_2 = id_index_t[index_2]
            index_1 = id_list_t.index(id_1)
            index_2 = id_list_t.index(id_2)
            edges_t += "{0} {1} \n".format(index_1, index_2)
    f.close()

    ## 保存源域图和目标域图
    f_s = open(output_source_dir, 'w', encoding='utf-8')
    f_t = open(output_target_dir, 'w', encoding='utf-8')
    f_s.write(edges_s)
    f_s.close()
    f_t.write(edges_t)
    f_t.close()
    print("Source graph and target graph have been saved!")

if __name__ == "__main__":
    args = parse_args()
    data_processing(args)