import numpy as np
import torch
import json
from scipy import sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import math
import ipdb
'''
class singleGCNLayer(nn.Module):

    def __init__(self, embsize, outsize, nolinear="NO"):
        super(singleGCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(embsize, outsize))
        self.nolinear = nolinear
        self.embsize = embsize
        self.outsize = outsize
        stdv = 1./math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, embeddings, adj):
        n = embeddings.size()[0]
        ret = torch.sparse.mm(adj, embeddings).matmul(self.W)
        if self.nolinear == "ReLU":
            ret = F.relu(ret)
            return ret
        return ret
'''

class singleGCNLayer(nn.Module):

    def __init__(self, embsize, outsize, nolinear="NO"):
        super(singleGCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(embsize, outsize))
        self.nolinear = nolinear
        self.embsize = embsize
        self.outsize = outsize
        stdv = 1./math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, embeddings, adj):
        n = embeddings.size()[0]
        # print(type(adj), type(embeddings), type(self.W))
        ret = torch.mm(adj.float(), embeddings).matmul(self.W)
        if self.nolinear == "ReLU":
            ret = F.relu(ret)
        return ret

    def batch_forward(self, embeddings, adj):
        bsz = embeddings.size(0)
        ret = torch.bmm(adj.float(), embeddings).bmm(self.W.unsqueeze(0).repeat(bsz, 1, 1))
        if torch.isnan(ret).any():
            ipdb.set_trace()
        if self.nolinear == "ReLU":
            ret = F.relu(ret)
        if torch.isnan(ret).any():
            ipdb.set_trace()
        return ret

class GCN(nn.Module):

    def __init__(self, embsize, hidsize, nclass, dropout):
        super(GCN, self).__init__()
        self.layer1 = singleGCNLayer(embsize, hidsize, "ReLU")
        self.layer2 = singleGCNLayer(hidsize, nclass)
        self.dropout = dropout

    def forward(self, embeddings, adj):
        n = embeddings.size()[0]
        embeddings = self.layer1(embeddings, adj)
        embeddings = F.dropout(embeddings, self.dropout, training=self.training) 
        embeddings = self.layer2(embeddings, adj)
        return embeddings

    def batch_forward(self, embeddings, adj):
        bsz = embeddings.size(0)
        assert bsz == adj.size(0)
        embeddings = self.layer1.batch_forward(embeddings, adj)
        if torch.isnan(embeddings).any():
            ipdb.set_trace()
        embeddings = F.dropout(embeddings, self.dropout, training=self.training) 
#         if torch.isnan(embeddings).any():
#             ipdb.set_trace()
        embeddings = self.layer2.batch_forward(embeddings, adj)
#         if torch.isnan(embeddings).any():
#             ipdb.set_trace()
        return embeddings

