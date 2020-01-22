## Reference
# Based on
#   https://github.com/rusty1s/pytorch_geometric/blob/master/examples/enzymes_topk_pool.py
#   Which is based on
#   Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks
#     https://arxiv.org/pdf/1810.02244.pdf
# Reference for TopK
#   Towards Sparse Hierarchical Graph Classifiers
#   https://arxiv.org/pdf/1811.01287.pdf
# Reference for TagConv
#   TOPOLOGY ADAPTIVE GRAPH CONVOLUTIONAL NETWORKS
#   https://arxiv.org/pdf/1710.10370.pdf


import torch
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F

from torch_geometric.nn import TAGConv
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gap#, TopKPooling

from custom_layers.topk_pool import TopKPooling as CustomTopK
from custom_layers.gcn_conv import GCNConv as CustomGCN

class Net(torch.nn.Module):
    def __init__(self, k=2, preconv_ws = [32,32,32],
                highway_layers=3, highway_w=64, lin_ws=[32,8]):
        super(Net, self).__init__()

        # Remove not active nodes
        self.conv = CustomGCN(2,1, cached=False)
        self.conv.weight.requires_grad = False

        self.topk = CustomTopK(1, min_score=0.1)
        self.topk.weight.requires_grad = False

        prev_w = 2

        # Pre Convolutions
        self.pre_convs = []
        self.pre_bns = []
        for i, w in enumerate(preconv_ws):
            setattr(self, "pre_conv{}".format(i), TAGConv(prev_w, w, k))
            setattr(self, "pre_bn{}".format(i), BatchNorm1d(w))
            self.pre_convs.append(getattr(self, "pre_conv{}".format(i)))
            self.pre_bns.append(getattr(self, "pre_bn{}".format(i)))
            prev_w = w

        # Highway Convolutions
        self.high_convs = []
        self.high_bns = []
        for i in range(highway_layers):
            setattr(self, "high_conv{}".format(i), TAGConv(prev_w, highway_w, k))
            setattr(self, "high_bn{}".format(i), BatchNorm1d(highway_w))
            self.high_convs.append(getattr(self, "high_conv{}".format(i)))
            self.high_bns.append(getattr(self, "high_bn{}".format(i)))
            prev_w = highway_w

        # MLP
        prev_w *= highway_layers
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

        for bn, conv in zip(self.pre_bns, self.pre_convs):
            x = F.elu(conv(x,edge_index))
            x = bn(x)

        xs = []
        for bn, conv in zip(self.high_bns, self.high_convs):
            x = F.elu(conv(x,edge_index))
            x = bn(x)
            # xs.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
            xs.append(gmp(x, batch_index))

        x = torch.cat(xs, dim=1)
        #x = xs[-1]

        for lin in self.lins:
            x = F.dropout(F.elu(lin(x)), 0.5, training=self.training)
            #x = F.elu(lin(x))

        x = self.lin_final(x)
        return F.log_softmax(x, dim=1)

