##########################################################################
# @File: initial_data_processing_cross.py
# @Author: Zhehan Liang
# @Date: 8/18/2020
# @Intro: Adjust the original data, process the number of nodes, the minimum degree of the deleted edge node,
# the data format, etc., and delete a certain number of non-duplicated edges respectively,
# and finally save them as two new graphs of the source domain and the target domain, and save the format As .txt
# tips: Cross network, e.g. flickr-lastfm
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
    ## Parameter setting
    datasets = args.dataset.split('-')
    dataset_s = datasets[0] # Source dataset
    dataset_t = datasets[1] # Target dataset
    input_dir = "./data/graph_data/" # Original data file path
    map_dir = "./data/graph_map/" # Map file path
    output_dir = "./data/graph_edge/" # New data file storage path

    ## Path setting
    input_data_edge_dir_s = input_dir + "{0}/{0}.edges".format(dataset_s) # Source domain graph edge path
    input_data_edge_dir_t = input_dir + "{0}/{0}.edges".format(dataset_t) # Target domain graph edge path
    input_data_node_dir_s = input_dir + "{0}/{0}.nodes".format(dataset_s) # Source domain graph node path
    input_data_node_dir_t = input_dir + "{0}/{0}.nodes".format(dataset_t) # Target domain graph node path
    map_file = map_dir + "{0}.map.raw".format(args.dataset) # Map file path
    output_source_dir = output_dir + "{0}_source_edges.txt".format(args.dataset) # File path of source domain graph
    output_target_dir = output_dir + "{0}_target_edges.txt".format(args.dataset) # File path of target domain graph

    ## Read the mapping file and store the id of each row as two lists in order
    ## The index of the id in the list is the index corresponding to the new map
    id_list_s = []
    id_list_t = []
    map_num = 0
    f = open(map_file, 'r', encoding='utf-8')
    for line in f.readlines():
        ids = line.strip().split(' ')
        assert len(ids)==2, "ids's length error!"
        id_list_s.append(ids[0])
        id_list_t.append(ids[1])
        map_num += 1
    f.close()

    ## Determine the index of each id in the original graph according to the '.nodes' file
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

    ## Traverse the .edges file, and if the two edge indexes of a row are keywords in the id_index dictionary,
    ## determine their indexes in the new graph according to the corresponding value (id name) and id_list,
    ## and record the edges that constitute it
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

    ## Save source and target graphs
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