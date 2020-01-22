import numpy as np
import torch
import json
from scipy import sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import math

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


