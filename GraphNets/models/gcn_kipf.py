## Reference
# SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS
#  https://arxiv.org/pdf/1609.02907.pdf
# With Addition
#   Add Global Pooling
#   Add linear predictor


import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool

class Net(torch.nn.Module):
    def __init__(self, w1=8, w2=32, w3=128):
        super(Net, self).__init__()
        self.conv1 = GCNConv(2, w1, cached=False)
        self.conv2 = GCNConv(w1, w2, cached=False)
        self.conv3 = GCNConv(w2, w3, cached=False)
        self.linear = Linear(w3, 3)

    def forward(self, batch):
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = global_max_pool(x, batch_index)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
