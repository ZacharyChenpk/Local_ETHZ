import json
import numpy as np
import scipy
import scipy.sparse as sp
import torch
import pickle
import sys
import networkx as nx

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str):

    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances(a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../GCNdata/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pickle.load(f, encoding='latin1'))
            else:
                objects.append(pickle.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../GCNdata/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).todense()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

# def normalize(adj, is_feature=False):
#     if not is_feature:
#         adj = adj + adj.T.multiply(adj.T > adj) - adj.T.multiply(adj.T > adj)
#         n = adj.shape[0]
#         adj = adj + sp.diags([1]*n)
#     rowsum = np.array(adj.sum(1))
#     dsqrt = np.power(rowsum, -0.5).reshape(-1)
#     dsqrt[np.isinf(dsqrt)] = 0.
#     dsqrt = sp.diags(dsqrt)
#     return sp.coo_matrix(dsqrt.dot(adj).dot(dsqrt))

def normalize(adj):
    n = adj.size(0)
    adj = adj + torch.eye(n)
    rowsum = torch.sum(adj, dim=1)
    rowsum = rowsum + 1e-8
    dsqrt = rowsum.rsqrt()
    infmsk = torch.ones_like(dsqrt, requires_grad=False).cuda()
    infmsk[torch.isinf(dsqrt)] = 0.
    infmsk[torch.isnan(dsqrt)] = 0.
    dsqrt = torch.diag(dsqrt.mul(infmsk))
    return dsqrt.mm(adj).mm(dsqrt)

def batch_normalize(adj):
    bsz = adj.size(0)
    n = adj.size(1)
    adj = adj + torch.eye(n).repeat(bsz,1,1)
    rowsum = torch.sum(adj, dim=2)
    rowsum = rowsum + 1e-8
    dsqrt = rowsum.rsqrt()
    infmsk = torch.ones_like(dsqrt, requires_grad=False).cuda()
    infmsk[torch.isinf(dsqrt)] = 0.
    infmsk[torch.isnan(dsqrt)] = 0.
    dsqrt = dsqrt.mul(infmsk).repeat(1,1,n).mul(torch.eye(n).repeat(bsz,1,1))
    return dsqrt.bmm(adj).bmm(dsqrt)

# def feature_norm(f):
#     rowsum = np.array(f.sum(1))
#     r_inv = np.power(rowsum, -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = sp.diags(r_inv)
#     f = r_mat_inv.dot(f)
#     return f

def feature_norm(f):
    rowsum = torch.sum(f, dim=1)
    rowsum = rowsum + 1e-8
    r_inv = torch.reciprocal(rowsum)
    infmsk = torch.ones_like(r_inv, requires_grad=False).cuda()
    infmsk[torch.isinf(r_inv)] = 0.
    infmsk[torch.isnan(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv.mul(infmsk)).cuda()
    f = r_mat_inv.mm(f)
    return f

def batch_feature_norm(f):
    bsz = f.size(0)
    n = f.size(1)
    rowsum = torch.sum(f, dim=2)
    rowsum = rowsum + 1e-8
    r_inv = torch.reciprocal(rowsum)
    infmsk = torch.ones_like(r_inv, requires_grad=False).cuda()
    infmsk[torch.isinf(r_inv)] = 0.
    infmsk[torch.isnan(r_inv)] = 0.
    r_mat_inv = r_inv.mul(infmsk).unsqueeze(2).repeat(1,1,n).mul(torch.eye(n).cuda().unsqueeze(0).repeat(bsz,1,1))
    f = r_mat_inv.bmm(f)
    return f

def cal_accuracy(output, y_val, val_mask):
    output = output[val_mask]
    y_val = y_val[val_mask]
    out_label = torch.argmax(output, dim=1)
    return sum([y_val[i][out_label[i]] for i in range(y_val.shape[0])])/float(y_val.shape[0])
