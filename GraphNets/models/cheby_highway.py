## Reference
# Based on
#   https://github.com/rusty1s/pytorch_geometric/blob/master/examples/enzymes_topk_pool.py
#   Which is based on
#   Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks
#     https://arxiv.org/pdf/1810.02244.pdf
# Reference for TopK
#   Towards Sparse Hierarchical Graph Classifiers
#   https://arxiv.org/pdf/1811.01287.pdf
# Reference for Cheby
#   Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering
#   https://arxiv.org/pdf/1606.09375.pdf


import torch
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F

from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gap#, TopKPooling

from custom_layers.topk_pool import TopKPooling as CustomTopK
from custom_layers.gcn_conv import GCNConv as CustomGCN

class Net(torch.nn.Module):
    def __init__(self, k=2, layers=3, graph_w=128, lin_ws=[32,8]):
        super(Net, self).__init__()

        # Remove not active nodes
        self.conv = CustomGCN(2,1, cached=False)
        self.conv.weight.requires_grad = False

        self.topk = CustomTopK(1, min_score=0.1)
        self.topk.weight.requires_grad = False

        # Convolutions
        prev_w = 2
        self.convs = []
        self.bns = []
        for i in range(layers):
            setattr(self, "conv{}".format(i), ChebConv(prev_w, graph_w, k))
            setattr(self, "bn{}".format(i), BatchNorm1d(graph_w))
            self.convs.append(getattr(self, "conv{}".format(i)))
            self.bns.append(getattr(self, "bn{}".format(i)))
            prev_w = graph_w

        # MLP
        prev_w *= layers
        self.lins = []
        for i, w in enumerate(lin_ws):
            setattr(self, "lin{}".format(i), Linear(prev_w, w))
            self.lins.append(getattr(self, "lin{}".format(i)))
            prev_w = w

        # Final Layer
        self.lin_final = Linear(prev_w, 3)

    def forward(self, batch):
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch

        attn = self.conv(x,edge_index)
        x, edge_index, _, batch_index, _, _ = self.topk(x, edge_index, None, batch_index, attn)

        xs = []
        for bn, conv in zip(self.bns, self.convs):
            x = F.elu(conv(x,edge_index))
            x = bn(x)
            # xs.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
            xs.append(gmp(x, batch_index))

        x = torch.cat(xs, dim=1)
        #x = xs[-1]

        for lin in self.lins:
            #x = F.dropout(F.elu(lin(x)), 0.5, training=self.training)
            x = F.elu(lin(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = self.lin_final(x)
        return F.log_softmax(x, dim=1)

