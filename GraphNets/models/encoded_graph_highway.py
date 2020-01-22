## Reference
# Based on
#   https://github.com/rusty1s/pytorch_geometric/blob/master/examples/enzymes_topk_pool.py
#   Which is based on
#   Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks
#     https://arxiv.org/pdf/1810.02244.pdf
#   With additional pre encoding layer
# Reference for TopK
#   Towards Sparse Hierarchical Graph Classifiers
#   https://arxiv.org/pdf/1811.01287.pdf
# Reference for Cheby
#   Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering
#   https://arxiv.org/pdf/1606.09375.pdf


import torch
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F

from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gap#, TopKPooling

class Net(torch.nn.Module):
    def __init__(self, layers=7, graph_w=128, concat_layers=[3,5,7], lin_ws=[32,8]):
        super(Net, self).__init__()

        self.concat_layers = concat_layers

        # Encoding
        self.pre_lin = Linear(38, 3)
        self.pre_bn = BatchNorm1d(3)

        # Convolutions
        prev_w = 3
        self.convs = []
        self.bns = []
        for i in range(layers):
            setattr(self, "conv{}".format(i), GraphConv(prev_w, graph_w))
            setattr(self, "bn{}".format(i), BatchNorm1d(graph_w))
            self.convs.append(getattr(self, "conv{}".format(i)))
            self.bns.append(getattr(self, "bn{}".format(i)))
            prev_w = graph_w

        # MLP
        prev_w *= len(concat_layers)
        self.lins = []
        for i, w in enumerate(lin_ws):
            setattr(self, "lin{}".format(i), Linear(prev_w, w))
            self.lins.append(getattr(self, "lin{}".format(i)))
            prev_w = w

        # Final Layer
        self.lin_final = Linear(prev_w, 3)

    def forward(self, batch):
        x, edge_index, batch_index = batch.x, batch.edge_index, batch.batch

        x = self.pre_bn(F.elu(self.pre_lin(x)))

        xs = []
        for bn, conv in zip(self.bns, self.convs):
            x = F.elu(conv(x,edge_index))
            x = bn(x)
            # xs.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
            xs.append(gmp(x, batch_index))

        x = torch.cat([xs[i-1] for i in self.concat_layers], dim=1)
        #x = xs[-1]

        for lin in self.lins:
            #x = F.dropout(F.elu(lin(x)), 0.5, training=self.training)
            x = F.elu(lin(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = self.lin_final(x)
        return F.log_softmax(x, dim=1)

