## Reference
# Based on
# SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS
# https://arxiv.org/pdf/1609.02907.pdf
# With modification
#   Replace Dropout with Batch Normalization
#   Add Global Max Pooling
#   Add linear predictor


import torch
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool

class Net(torch.nn.Module):
    def __init__(self, w1=8, w2=32, w3=128):
        super(Net, self).__init__()
        self.conv1 = GCNConv(2, w1, cached=False)
        self.bn1 = BatchNorm1d(w1)
        self.conv2 = GCNConv(w1, w2, cached=False)
        self.bn2 = BatchNorm1d(w2)
        self.conv3 = GCNConv(w2, w3, cached=False)
        self.bn3 = BatchNorm1d(w3)
        self.linear = Linear(w3, 3)

    def forward(self, batch):
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = global_max_pool(x, batch_index)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
