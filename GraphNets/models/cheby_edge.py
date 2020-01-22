## Reference
# Based on
# SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS
# https://arxiv.org/pdf/1609.02907.pdf
# Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering
# https://arxiv.org/pdf/1606.09375.pdf
# Towards Graph Pooling by Edge Contraction
# https://graphreason.github.io/papers/17.pdf
# With modification
#   Use Chebyshev Convolution instead of Graph convolution by kipf
#   Use TopK to remove non activated nodes to save memory
#   Replace Dropout with Batch Normalization
#   Add edge pooling to shrink graph
#   Add Global Max Pooling
#   Add linear predictor


import torch
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F

from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_max_pool, EdgePooling

from custom_layers.topk_pool import TopKPooling
from custom_layers.gcn_conv import GCNConv as CustomGCN
#from custom_layers.remove_isolated_nodes import remove_isolated_nodes

class Net(torch.nn.Module):
    def __init__(self, k=2, w1=128, w2=128, w3=128):
        super(Net, self).__init__()

        self.conv = CustomGCN(2,1, cached=False)
        self.conv.weight.requires_grad = False

        self.topk = TopKPooling(1, min_score=0.1)
        self.topk.weight.requires_grad = False

        self.conv1 = ChebConv(2, w1, k)
        self.bn1 = BatchNorm1d(w1)
        self.pool1 = EdgePooling(w1)

        self.conv2 = ChebConv(w1, w2, k)
        self.bn2 = BatchNorm1d(w2)
        self.pool2 = EdgePooling(w2)

        self.conv3 = ChebConv(w2, w3, k)
        self.bn3 = BatchNorm1d(w3)

        self.linear = Linear(w3, 3)



    def forward(self, batch):
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch

        attn = self.conv(x,edge_index)
        x, edge_index, _, batch_index, _, _ = self.topk(x, edge_index, None, batch_index, attn)

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x, edge_index, batch_index, _ = self.pool1(x, edge_index, batch_index)

        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x, edge_index, batch_index, _ = self.pool2(x, edge_index, batch_index)

        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)

        x = global_max_pool(x, batch_index)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
