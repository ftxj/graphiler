import torch.nn as nn
import torch.nn.functional as F
import torch
from dgl.nn.pytorch.conv import GATConv

from graphiler.utils import load_data, setup, check_equal, bench, homo_dataset, DEFAULT_DIM, init_log, empty_cache


class GAT_DGL(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GAT_DGL, self).__init__()
        print(type(in_dim))
        print(in_dim)
        self.layer1 = GATConv(in_dim, hidden_dim,
                              num_heads=1, allow_zero_in_degree=True)
        self.layer2 = GATConv(hidden_dim, out_dim,
                              num_heads=1, allow_zero_in_degree=True)

    def forward(self, g, features):
        print("DGL forward")
        print(features.size())
        h = self.layer1(g, features)
        h = F.elu(h)
        h = self.layer2(g, h)
        return h



# device = setup()

# feat_dim = 500
# hidden_dim = 32
# out_dim = 12

# net_dgl = GAT_DGL(in_dim=feat_dim, hidden_dim=DEFAULT_DIM, out_dim=DEFAULT_DIM).to(device)


# print("DGL Graph")
# print(torch.jit.script(net_dgl).graph)
# print("DGL Inlined Graph")
# print(torch.jit.script(net_dgl).inlined_graph)
