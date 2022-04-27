import dgl
import torch as th
from torch import nn


dataset = dgl.data.PubmedGraphDataset()

graph = dataset[0]

print(graph)

feat = torch.rand([g.number_of_nodes(), 32])



num_heads = 2
out_feats = 8

feat_drop = nn.Dropout(0)

attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))

fc = nn.Linear(32, out_feats * num_heads, bias=False)

leaky_relu =  nn.LeakyReLU(0.2)

if isinstance(feat, tuple):
    h_src = feat_drop(feat[0])
    h_dst = feat_drop(feat[1])
else:
    h_src = h_dst = feat_drop(feat)
    feat_src = feat_dst = fc(h_src).view(-1, num_heads, out_feats)
    if graph.is_block:
        feat_dst = feat_src[:graph.number_of_dst_nodes()]

el = (feat_src * attn_l).sum(dim=-1).unsqueeze(-1)

er = (feat_dst * attn_r).sum(dim=-1).unsqueeze(-1)

graph.srcdata.update({'ft': feat_src, 'el': el})

graph.dstdata.update({'er': er})

# compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.

graph.apply_edges(fn.u_add_v('el', 'er', 'e'))


e = leaky_relu(graph.edata.pop('e'))