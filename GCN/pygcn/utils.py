# 如果某个版本中出现了某个新的功能特性，而且这个特性和当前版本中使用的不兼容，
# 也就是说它在当前版本中不是语言标准，那么我们如果想要使用的话就要从__future__模块导入
from __future__ import print_function       # print()函数

import numpy as np  # python中操作数组的函数
import scipy.sparse as sp   # python中稀疏矩阵相关库
import torch

label = [99, 76, 6, 4, 5, 5, 4, 52, 3]


def encode_onehot(lables):
    """
    将labels转换成onehot向量
    :param lables:
    :return:
    """
    # set集合，消除重复的元素.没有排序功能，乱序排列
    classes = set(lables)
    # 用单位矩阵来构建onehot向量,np.identity创建一个单位方阵
    # i是下标，c是set中的元素
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    # classes_dict.get从字典中获取值,
    # map将lables中的元素和classes_dict对应起来，返回地址，然后使用list返回列表，
    # 再用np.array返回数组矩阵 ；dtype将浮点数转换成小数
    lable_onehot = np.array(list(map(classes_dict.get, lables)),
                            dtype=np.int32)

    return lable_onehot


def onehottolabels(labels):
    """
    将onehot转换成label
    :param labels:
    :return:
    """
    # np.where(labels)[1]返回非零位置的下标
    return torch.LongTensor(np.where(labels)[1])


def load_data(path="../data/cora/", dataset="cora"):
    """
    下载cora数据集
    :param path:
    :param dataset:
    :return:
    """
    # str.format()函数用于格式化字符串
    print('Loading {} dataset...'.format(dataset))
    # np.genfromtxt()函数用于从.csv文件或.tsv文件中生成数组
    # np.genfromtxt(fname, dtype, delimiter, usecols, skip_header)
    # frame：文件名
    # dtype：数据类型
    # delimiter：分隔符
    # usecols：选择读哪几列，通常将属性集读为一个数组，将标签读为一个数组
    # skip_header：是否跳过表头
    # genfromtxt函数创建数组表格数据
    # genfromtxt主要执行两个循环运算。第一个循环将文件的每一行转换成字符串序列。
    # 第二个循环将每个字符串序列转换为相应的数据类型
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    # 提取样本的特征，并将其转换为csr矩阵（压缩稀疏行矩阵），用行索引、列索引和值表示矩阵
    # idx_features_labels[:, 1:-1]不包括最后一列
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 提取样本的标签，并将其转换为one-hot编码形式
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    # 样本的id数组
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 有样本id到样本索引的映射字典
    idx_map = {j: i for i, j in enumerate(idx)}
    # 样本之间的引用关系数组
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    # 将样本之间的引用关系用样本索引之间的关系表示
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    # 构建图的邻接矩阵，用坐标形式的稀疏矩阵表示，非对称邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    # 将非对称邻接矩阵转变为对称邻接矩阵,multiply表示对应元素相乘
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))


    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # 特征的密集矩阵表示
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize(adj, symmetric=False):
    # 如果邻接矩阵为对称矩阵，得到对称归一化邻接矩阵
    # D^(-1/2) * A * D^(-1/2)
    if symmetric:
        # A.sum(axis=1)：计算矩阵的每一行元素之和，得到节点的度矩阵D
        # np.power(x, n)：数组元素求n次方，得到D^(-1/2)
        # sp.diags()函数根据给定的对象创建对角矩阵，对角线上的元素为给定对象中的元素
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        # tocsr()函数将矩阵转化为压缩稀疏行矩阵
        a_norm = d.dot(adj).dot(d).tocsr()
    # 如果邻接矩阵不是对称矩阵，得到随机游走正则化拉普拉斯算子
    # D^(-1) * A
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct/len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    将稀疏矩阵转换成稀疏张量表示
    :param sparse_mx:
    :return:
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

adj, features, labels, idx_train, idx_val, idx_test = load_data()

# adj = np.array([[1, 0, 1, 1], [0,1,1,0],[1,1,0,1],[1,0,1,0]])
# print(adj)
# d = np.array(adj.sum(1))
# print(d)
# d = sp.diags(np.power(d, -0.5).flatten(), 0)
# print(d)
# spaea = np.matrix([[1,0,0,0,0], [1,0,1,0,0],[0,0,0,1,0],[0,1,0,0,0],[1,1,0,1,0]], dtype=np.float32)
# sp = normalize(spaea)
# d = sparse_mx_to_torch_sparse_tensor(spaea)
# print(d)



