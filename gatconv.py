import dgl
import torch
from torch import nn


dataset = dgl.data.PubmedGraphDataset()

graph = dataset[0]

print(graph)

feat = torch.rand([graph.number_of_nodes(), 32])



num_heads = 2
out_feats = 8

feat_drop = nn.Dropout(0)
atten_drop = nn.Dropout(0)

attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))

fc = nn.Linear(32, out_feats * num_heads, bias=False)

leaky_relu =  nn.LeakyReLU(0.2)

if isinstance(feat, tuple):
    h_src = feat_drop(feat[0])
    h_dst = feat_drop(feat[1])
else:
    h_src = h_dst = feat_drop(feat)
    print('h_src : ', h_src.type(), h_src.size())
    
    print('h_dst : ', h_dst.type(), h_dst.size())
    
    feat_src = feat_dst = fc(h_src).view(-1, num_heads, out_feats)

    print('feat_src : ', feat_src.type(), feat_src.size())

    print('feat_dst : ', feat_dst.type(), feat_dst.size())

    if graph.is_block:
        feat_dst = feat_src[:graph.number_of_dst_nodes()]

el = (feat_src * attn_l).sum(dim=-1).unsqueeze(-1)

print('el : ', el.type(), el.size())

er = (feat_dst * attn_r).sum(dim=-1).unsqueeze(-1)

print('er : ', er.type(), er.size())

graph.srcdata.update({'ft': feat_src, 'el': el})

print(graph.srcdata)

graph.dstdata.update({'er': er})

print(graph.dstdata)

# compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.

func = dgl.function.u_add_v('el', 'er', 'e')


alldata = [graph.srcdata, graph.dstdata, graph.edata]

print('lhs', func.lhs)
print('lhs field', func.lhs_field)


if isinstance(func, dgl.function.BinaryMessageFunction):
    
    x = alldata[func.lhs][func.lhs_field]
    y = alldata[func.rhs][func.rhs_field]
    print(x.size())
    print(y.size())
    print(func.name)



graph.apply_edges(dgl.function.u_add_v('el', 'er', 'e'))


e = leaky_relu(graph.edata.pop('e'))



graph.edata['a'] = attn_drop(dgl.functional.edge_softmax(graph, e))

# message passing
graph.update_all(dgl.function.u_mul_e('ft', 'a', 'm'), dgl.function.sum('m', 'ft'))

rst = graph.dstdata['ft']
