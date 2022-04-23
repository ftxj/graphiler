import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
import torch

from graphiler.utils import setup

class GCN_PyG(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim):
        super(GCN_PyG, self).__init__()

        self.layer1 = GCNConv(in_dim, hidden_dim, cached=False).jittable()
        self.layer2 = GCNConv(hidden_dim, out_dim, cached=False).jittable()

    def forward(self, x, adj):
        h = self.layer1(x, adj)
        h = F.relu(h)
        h = self.layer2(h, adj)
        return h

device = setup()

feat_dim = 500
hidden_dim = 32
out_dim = 12

net_dgl = GCN_PyG(in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=out_dim).to(device)

script_model = torch.jit.script(net_dgl)

print("DGL Graph")
print(script_model.graph)
print("DGL Inlined Graph")
print(script_model.inlined_graph)