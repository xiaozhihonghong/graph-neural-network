import torch
import torch.nn as nn
import numpy as np

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs, num_sample=4):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        print(nodes[:10], '*'*100)
        if num_sample is not None:
            samp_neighs = [set(random.sample(to_neigh, num_sample))
                           if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
            print('samp_neighs=', len(samp_neighs))
            # random.sample从to_neigh中随机获取num_sample个邻居，不够就直接取邻居
        else:
            samp_neighs = to_neighs

        if self.gcn:
            # noinspection PySetFunctionToLiteral
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}   # 给节点标号
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = torch.div(mask, num_neigh)
        print('mask=', mask.shape)
        print('-----------------------------------------------------------------')
        if self.cuda:
            # noinspection PyArgumentList
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

            # noinspection PyArgumentList
            print(self.features, '*-'*50)
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))

            print('=======================================================')

            print('embed_matrix=', embed_matrix.size())
        print('mask2=', mask.shape)
        to_feats = mask.mm(embed_matrix)
        print('to_feat=', to_feats.shape)
        return to_feats
