##########################################################################
# @File: utils.py
# @Author: Zhehan Liang
# @Date: 6/8/2020
# @Intro: 一些需要用到的工具函数
##########################################################################

import os
import numpy as np
import networkx as nx
import heapq as hq
import torch
import time
import sklearn.metrics.pairwise

def adjacency_normalize(A , symmetric=True):
    """
    根据GCN的公式对邻接矩阵进行标准化
    """
    # A = A+I
    A = A + torch.eye(A.size(0)) # 邻接矩阵对角线为0时需要
    # 计算所有节点的度
    d = A.sum(1)
    if symmetric:
        #D = D^-1/2
        D = torch.diag(torch.pow(d , -0.5))
        return D.mm(A).mm(D)
    else :
        # D=D^-1
        D =torch.diag(torch.pow(d,-1))
        return D.mm(A)

def read_graph(params):
    """
    读取图
    """
    time_head = time.time() # 记录开始时间
    source_graph_path = params.data_path + params.dataset + "_source_edges.txt"
    target_graph_path = params.data_path + params.dataset + "_target_edges.txt"
    g_source = nx.read_edgelist(source_graph_path, nodetype=int)
    g_target = nx.read_edgelist(target_graph_path, nodetype=int)
    time_tail = time.time() # 记录完成时间
    print("=====> Graphs have been read!\tTime: %.3f"%(time_tail-time_head))

    return g_source, g_target

def adjacency_matrix_normalize(params, G_source, G_target):
    """
    读取图的邻接矩阵
    """
    time_head = time.time() # 记录开始时间
    A_source = nx.adjacency_matrix(G_source)
    A_target = nx.adjacency_matrix(G_target)
    time_tail = time.time() # 记录完成时间
    print("=====> Adjacency matrixs have been read!\tTime: %.3f"%(time_tail-time_head))
    time_head = time.time() # 记录开始时间
    A_source_norm = adjacency_normalize(torch.FloatTensor(A_source),True)
    A_target_norm = adjacency_normalize(torch.FloatTensor(A_target),True)
    # # 报错:sparse matrix length is ambiguous 时添加.todensce()
    # A_source_norm = adjacency_normalize(torch.FloatTensor(A_source.todense()),True)
    # A_target_norm = adjacency_normalize(torch.FloatTensor(A_target.todense()),True)
    time_tail = time.time() # 记录完成时间
    print("=====> Adjacency matrixs have been normalized!\tTime: %.3f"%(time_tail-time_head))
    if params.cuda:
        return A_source_norm.cuda(), A_target_norm.cuda()
    else:
        return A_source_norm, A_target_norm

def get_node_num(params):
    """
    从文件名读取节点数目
    """
    path_list = os.listdir(params.data_path) # 获取数据路径下的文件名
    for file_name in path_list:
        if ".edge" in file_name and params.dataset in file_name: # 找到数据集对应的.edge文件
            num = int(file_name.split("_")[1]) # 解析文件名，文件名中第一个_后的数字就是节点数量
            print("=====> number of nodes: %d"%(num))
            return(num)

def mean_metric(metric_matrix):
    """
    计算平均相似度
    """
    mm1 = (metric_matrix.sum(axis=1)/metric_matrix.shape[0])
    mm2 = (metric_matrix.sum(axis=0)/metric_matrix.shape[1])
    return mm1, mm2

def get_metric_matrix(matrix1, matrix2, method):
    """
    计算度量矩阵
    """
    assert method in ['Euclid', 'cosine'], "Unkown operation!" # 校验度量方式
    if method=='cosine':
        metric_matrix = sklearn.metrics.pairwise.cosine_similarity(matrix1, matrix2)
        ## 不使用库的计算方法
        # dot = matrix1.dot(matrix2.transpose())
        # matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))
        # matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum(axis=1))
        # metric_matrix = np.divide(dot, matrix1_norm * matrix2_norm.transpose())
    else:
        metric_matrix = sklearn.metrics.pairwise.euclidean_distances(matrix1, matrix2)
        ## 不使用库的计算方法
        # metric_matrix = np.sqrt(-2*np.dot(matrix1, matrix2.T) + np.sum(np.square(matrix2), axis = 1) + np.transpose([np.sum(np.square(matrix1), axis = 1)]))
    mm_1 , mm_2 = mean_metric(metric_matrix)
    metric_matrix = metric_matrix - (np.tile(mm_1, (len(matrix1), 1)).T + np.tile(mm_2, (len(matrix1), 1))) * 0.4 # 再计算CGSS方法下的度量矩阵
    if method=='cosine': # cosine情况下需要进行对数操作来保证数值越低越好
        metric_matrix = np.exp(-metric_matrix)

    return metric_matrix

def get_two_graph_scores(embeddings1, embeddings2, operation, method):
    """
    计算两个图之间节点在不同top值下的匹配准确率/计算最佳匹配点
    """
    assert embeddings1.shape==embeddings2.shape, "embeddings1.shape!=embeddings2.shape" # 校验嵌入的形状是否满足要求，该方案中要求节点的数量相等
    assert operation in ['evaluate', 'match'], "Unkown operation!" # 校验操作是否是评估和匹配中的一个
    top1_count = 0
    top5_count = 0
    top10_count = 0
    pairs = []
    metric_matrix = get_metric_matrix(embeddings1, embeddings2, method)
    time_head = time.time() # 记录开始时间
    # print("=====> metric_matrix:")
    # print(metric_matrix)
    for i in range(len(embeddings1)):
        sort = np.argsort(metric_matrix[i])
        if operation=='evaluate': # 评估时，对top进行解析，得到各指标下匹配正确的数量
            if sort[0] == i: # 当时最佳匹配时，不考察后续
                top1_count += 1
                top5_count += 1
                top10_count += 1
            else:
                for num in range(10):
                    if num <5 and sort[num] == i:
                        top5_count += 1
                        top10_count += 1
                    elif sort[num] == i:
                        top10_count += 1
        else: # 匹配时，记录对应最佳匹配点
            pairs.append(sort[0])
        # 每1000个点打印一次结果
        if i % 1000 == 0:
            time_tail = time.time() # 记录完成时间
            print("=====> Have matched %d nodes\tTime: %.3f"%(i, time_tail-time_head))
            if operation=='evaluate':
                print("=====> Accuracy number: top-1 %d, top-5 %d, top-10 %d"%(top1_count, top5_count, top10_count))
            time_head = time.time() # 记录开始时间
    if operation=='evaluate':
        return top1_count/len(embeddings1), top5_count/len(embeddings1), top10_count/len(embeddings1)
    else:
        assert len(pairs)==len(embeddings1), "Length of pairs is error!" # 校验最佳匹配pair长度是否正确
        return pairs

def load_embeddings(params, source):
    """
    载入编码profile得到的初始特征矩阵
    """
    time_head = time.time() # 记录开始时间
    feature = np.ones(shape=(params.node_num, params.init_dim)) # 初始化嵌入矩阵
    emb_dir = params.emb_dir + params.dataset
    emb_dir += "_source.emb" if source else "_target.emb" # 设置嵌入文件路径
    with open(emb_dir, 'r', encoding='utf-8') as f: # 读取嵌入文件
        for i, line in enumerate(f.readlines()): # 逐行进行读取
            if i == 0: # deepwalk生成的嵌入文件第一行是节点数和嵌入维度
                result = line.strip().split(' ')
                # 校验嵌入信息
                assert len(result)==2, "Information format of embeddings is error!"
                assert int(result[0])==params.node_num, "Amount of embeddings is error!"
                assert int(result[1])==params.init_dim, "Dimension of embeddings is error!"
            else: # 第一行以后都是嵌入信息，格式为节点+对应的嵌入
                node, vector = line.strip().split(' ', 1) # 将单行文本分为节点和嵌入向量
                vector = np.fromstring(vector, sep=' ') # 将文本向量转化为numpy向量
                assert len(vector)==params.init_dim, "Length of embeddings is error!" # 校验嵌入长度是否正确
                feature[int(node)] = vector # 将得到的向量存入嵌入矩阵的对应行
    embeddings = torch.from_numpy(feature).float() # 把numpy转换到tensor
    embeddings = embeddings.cuda() if params.cuda else embeddings # cuda转化
    time_tail = time.time() # 记录完成时间
    if source:
        print("Source embeddings have been loaded!\tTime: %.3f"%(time_tail-time_head))
    else:
        print("Target embeddings have been loaded!\tTime: %.3f"%(time_tail-time_head))
    assert embeddings.size() == (params.node_num, params.init_dim), "Size of embeddings is error!" # 校验embedding的尺寸是否正确
    # 初始化cuda
    if params.cuda:
        embeddings.cuda()

    return embeddings
