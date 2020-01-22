import numpy as np
import torch
import json
from scipy import sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import sys
import networkx as nx
import os
import time

from utils import load_data, normalize, cal_accuracy
from model import GCN

datastr = "pubmed"

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(datastr)
#adj = nx.to_scipy_sparse_matrix(adj)
norm_adj = normalize(adj)
#norm_adj = adj

#print(adj.shape, adj_n.shape, features.shape, y_train.shape, train_mask.shape)
i = torch.LongTensor([norm_adj.row, norm_adj.col])
v = torch.FloatTensor(norm_adj.data)
norm_adj = torch.sparse.FloatTensor(i,v, adj.shape)
features = torch.tensor(features, dtype=torch.float, requires_grad=False)
y_train = torch.tensor(y_train, dtype=torch.long, requires_grad=False)
y_val = torch.tensor(y_val, dtype=torch.long, requires_grad=False)
y_test = torch.tensor(y_test, dtype=torch.long, requires_grad=False)

load_from = "gcn_model"
gcn = torch.load(load_from)
gcn.eval()
output = gcn(features, norm_adj)
test_acc = cal_accuracy(output, y_test, test_mask)
print("test acc:", test_acc)
