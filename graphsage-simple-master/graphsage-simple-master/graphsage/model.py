import torch
import torch.nn as nn
from torch.nn import init

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""


class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        # noinspection PyArgumentList
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        print('///////////////////////////////////')
        scores = self.weight.mm(embeds)
        print('scores=', scores.shape)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open('../cora/cora.content', 'r') as f:
        for i, line in enumerate(f):
            info = line.strip().split()  # 转换成列表
            feat_data[i, :] = np.array(list(map(float, info[1:-1])), dtype=np.float32)
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)    # 当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值
    with open('../cora/cora.cites', 'r') as f:
        for i, line in enumerate(f):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


def run_cora():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)  # 直接Embedding，2708个词，1433个维度，onehot编码
    # noinspection PyArgumentList
    features.weight = nn.parameter.Parameter(torch.FloatTensor(feat_data), requires_grad=False)  # 2708, 1433
    # features.cuda()

    agg1 = MeanAggregator(features, cuda=False)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=False, cuda=False, num_sample=4)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    # noinspection PyUnresolvedReferences
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=False, cuda=False, num_sample=4)

    # noinspection PyTypeChecker
    graphsage = SupervisedGraphSage(7, enc2)
    # graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)    # 对num_nodes随机排列
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.7)
    # 从graphsage.parameters()过滤出p.requires_grad所需的参数
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        # noinspection PyArgumentList
        print(batch_nodes[:10])
        # noinspection DuplicatedCode
        loss = graphsage.loss(batch_nodes, torch.LongTensor(labels[np.array(batch_nodes)]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item())

    val_output = graphsage.forward(val)
    print("Test F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

    test_output = graphsage.forward(test)
    print("Test F1:", f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

def load_pubmed():
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("../pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}  # 第一行里再次读取
        # split("\t")表示每行用tab键分割每一行
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("../pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_pubmed():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.parameter.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False, num_sample=10)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False, num_sample=25)

    graphsage = SupervisedGraphSage(3, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                torch.LongTensor(labels[np.array(batch_nodes)]))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item())

    val_output = graphsage.forward(val) 
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))

if __name__ == "__main__":
    run_cora()
