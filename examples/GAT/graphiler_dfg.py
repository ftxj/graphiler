def mpdfg_func(dglgraph: __torch__.torch.classes.my_classes.DGLGraph,
    ndata: Dict[str, Tensor],
    edata: Dict[str, Tensor],
    ntypedata: Dict[str, Tensor],
    etypedata: Dict[str, Tensor],
    fc_weight_msg: Tensor,
    attn_weight_msg: Tensor) -> Dict[str, Tensor]:
  _0 = torch.tensor(0, dtype=None, device=None, requires_grad=False)
  return {"h": _0}

UDF message function:
 graph(%edges.1 : __torch__.graphiler.dgl_udf_batch.EdgeBatchDummy,
      %fc_weight.1 : Tensor,
      %attn_weight.1 : Tensor):
  %3 : float = prim::Constant[value=0.01]()
  %4 : str = prim::Constant[value="e"]() # examples/GAT/GAT.py:55:22
  %5 : str = prim::Constant[value="z"]() # examples/GAT/GAT.py:55:12
  %6 : str = prim::Constant[value="h"]() # examples/GAT/GAT.py:51:29
  %7 : int = prim::Constant[value=1]() # examples/GAT/GAT.py:53:35
  %19 : Dict(str, Tensor) = prim::GetAttr[name="_src_data"](%edges.1)
  %9 : Tensor = aten::__getitem__(%19, %6) # examples/GAT/GAT.py:51:19
  %z_s.1 : Tensor = aten::mm(%9, %fc_weight.1) # examples/GAT/GAT.py:51:10
  %20 : Dict(str, Tensor) = prim::GetAttr[name="_dst_data"](%edges.1)
  %12 : Tensor = aten::__getitem__(%20, %6) # examples/GAT/GAT.py:52:19
  %z_d.1 : Tensor = aten::mm(%12, %fc_weight.1) # examples/GAT/GAT.py:52:10
  %14 : Tensor[] = prim::ListConstruct(%z_s.1, %z_d.1)
  %z2.1 : Tensor = aten::cat(%14, %7) # examples/GAT/GAT.py:53:9
  %a.1 : Tensor = aten::mm(%z2.1, %attn_weight.1) # examples/GAT/GAT.py:54:8
  %17 : Tensor = aten::leaky_relu_(%a.1, %3) # examples/GAT/GAT.py:55:27
  %18 : Dict(str, Tensor) = prim::DictConstruct(%5, %z_s.1, %4, %17)
  return (%18)

UDF reduce function:
 graph(%nodes.1 : __torch__.graphiler.dgl_udf_batch.NodeBatchDummy):
  %1 : str = prim::Constant[value="h"]() # examples/GAT/GAT.py:61:12
  %2 : bool = prim::Constant[value=0]()
  %3 : str = prim::Constant[value="z"]() # examples/GAT/GAT.py:60:40
  %4 : None = prim::Constant()
  %5 : str = prim::Constant[value="e"]() # examples/GAT/GAT.py:59:40
  %6 : int = prim::Constant[value=1]() # examples/GAT/GAT.py:59:50
  %16 : Dict(str, Tensor) = prim::GetAttr[name="_msgs"](%nodes.1)
  %8 : Tensor = aten::__getitem__(%16, %5) # examples/GAT/GAT.py:59:26
  %alpha.1 : Tensor = aten::softmax(%8, %6, %4) # examples/GAT/GAT.py:59:12
  %17 : Dict(str, Tensor) = prim::GetAttr[name="_msgs"](%nodes.1)
  %11 : Tensor = aten::__getitem__(%17, %3) # examples/GAT/GAT.py:60:26
  %12 : Tensor = aten::mul(%alpha.1, %11) # examples/GAT/GAT.py:60:18
  %13 : int[] = prim::ListConstruct(%6)
  %h.1 : Tensor = aten::sum(%12, %13, %2, %4) # examples/GAT/GAT.py:60:8
  %15 : Dict(str, Tensor) = prim::DictConstruct(%1, %h.1)
  return (%15)

UDF update function:
 None
Before Buildre MP-DFG:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %11 : bool = prim::Constant[value=0]()
  %9 : None = prim::Constant()
  %7 : str = prim::Constant[value="h"]() # /root/.dgl/mpdfg_temp.py:10:12
  %8 : int = prim::Constant[value=0]() # /root/.dgl/mpdfg_temp.py:10:29
  %12 : Tensor = aten::tensor(%8, %9, %9, %11) # /root/.dgl/mpdfg_temp.py:10:16
  %13 : Dict(str, Tensor) = prim::DictConstruct(%7, %12)
  return (%13)
Before Buildre Msg-DFG:
graph(%edges.1 : __torch__.graphiler.dgl_udf_batch.EdgeBatchDummy,
      %fc_weight.1 : Tensor,
      %attn_weight.1 : Tensor):
  %3 : float = prim::Constant[value=0.01]()
  %4 : str = prim::Constant[value="e"]() # examples/GAT/GAT.py:55:22
  %5 : str = prim::Constant[value="z"]() # examples/GAT/GAT.py:55:12
  %6 : str = prim::Constant[value="h"]() # examples/GAT/GAT.py:51:29
  %7 : int = prim::Constant[value=1]() # examples/GAT/GAT.py:53:35
  %19 : Dict(str, Tensor) = prim::GetAttr[name="_src_data"](%edges.1)
  %9 : Tensor = aten::__getitem__(%19, %6) # examples/GAT/GAT.py:51:19
  %z_s.1 : Tensor = aten::mm(%9, %fc_weight.1) # examples/GAT/GAT.py:51:10
  %20 : Dict(str, Tensor) = prim::GetAttr[name="_dst_data"](%edges.1)
  %12 : Tensor = aten::__getitem__(%20, %6) # examples/GAT/GAT.py:52:19
  %z_d.1 : Tensor = aten::mm(%12, %fc_weight.1) # examples/GAT/GAT.py:52:10
  %14 : Tensor[] = prim::ListConstruct(%z_s.1, %z_d.1)
  %z2.1 : Tensor = aten::cat(%14, %7) # examples/GAT/GAT.py:53:9
  %a.1 : Tensor = aten::mm(%z2.1, %attn_weight.1) # examples/GAT/GAT.py:54:8
  %17 : Tensor = aten::leaky_relu_(%a.1, %3) # examples/GAT/GAT.py:55:27
  %18 : Dict(str, Tensor) = prim::DictConstruct(%5, %z_s.1, %4, %17)
  return (%18)
Before Buildre Reduce-DFG:
graph(%nodes.1 : __torch__.graphiler.dgl_udf_batch.NodeBatchDummy):
  %1 : str = prim::Constant[value="h"]() # examples/GAT/GAT.py:61:12
  %2 : bool = prim::Constant[value=0]()
  %3 : str = prim::Constant[value="z"]() # examples/GAT/GAT.py:60:40
  %4 : None = prim::Constant()
  %5 : str = prim::Constant[value="e"]() # examples/GAT/GAT.py:59:40
  %6 : int = prim::Constant[value=1]() # examples/GAT/GAT.py:59:50
  %16 : Dict(str, Tensor) = prim::GetAttr[name="_msgs"](%nodes.1)
  %8 : Tensor = aten::__getitem__(%16, %5) # examples/GAT/GAT.py:59:26
  %alpha.1 : Tensor = aten::softmax(%8, %6, %4) # examples/GAT/GAT.py:59:12
  %17 : Dict(str, Tensor) = prim::GetAttr[name="_msgs"](%nodes.1)
  %11 : Tensor = aten::__getitem__(%17, %3) # examples/GAT/GAT.py:60:26
  %12 : Tensor = aten::mul(%alpha.1, %11) # examples/GAT/GAT.py:60:18
  %13 : int[] = prim::ListConstruct(%6)
  %h.1 : Tensor = aten::sum(%12, %13, %2, %4) # examples/GAT/GAT.py:60:8
  %15 : Dict(str, Tensor) = prim::DictConstruct(%1, %h.1)
  return (%15)
After Merge Message MP-DFG=:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %11 : bool = prim::Constant[value=0]()
  %9 : None = prim::Constant()
  %7 : str = prim::Constant[value="h"]() # /root/.dgl/mpdfg_temp.py:10:12
  %8 : int = prim::Constant[value=0]() # /root/.dgl/mpdfg_temp.py:10:29
  %12 : Tensor = aten::tensor(%8, %9, %9, %11) # /root/.dgl/mpdfg_temp.py:10:16
  %23 : float = prim::Constant[value=0.01]()
  %24 : str = prim::Constant[value="e"]()
  %25 : str = prim::Constant[value="z"]()
  %26 : str = prim::Constant[value="h"]()
  %27 : int = prim::Constant[value=1]()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %29 : Tensor = my_ops::BroadcastSrcNode(%28, %dglgraph)
  %z_s.1 : Tensor = aten::mm(%29, %fc_weight_msg)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %32 : Tensor = my_ops::BroadcastDstNode(%31, %dglgraph)
  %z_d.1 : Tensor = aten::mm(%32, %fc_weight_msg)
  %34 : Tensor[] = prim::ListConstruct(%z_s.1, %z_d.1)
  %z2.1 : Tensor = aten::cat(%34, %27)
  %a.1 : Tensor = aten::mm(%z2.1, %attn_weight_msg)
  %37 : Tensor = aten::leaky_relu_(%a.1, %23)
  %13 : Dict(str, Tensor) = prim::DictConstruct(%7, %12)
  return (%13)
After Build MP-DFG=:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %11 : bool = prim::Constant[value=0]()
  %9 : None = prim::Constant()
  %7 : str = prim::Constant[value="h"]() # /root/.dgl/mpdfg_temp.py:10:12
  %8 : int = prim::Constant[value=0]() # /root/.dgl/mpdfg_temp.py:10:29
  %12 : Tensor = aten::tensor(%8, %9, %9, %11) # /root/.dgl/mpdfg_temp.py:10:16
  %23 : float = prim::Constant[value=0.01]()
  %24 : str = prim::Constant[value="e"]()
  %25 : str = prim::Constant[value="z"]()
  %26 : str = prim::Constant[value="h"]()
  %27 : int = prim::Constant[value=1]()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %29 : Tensor = my_ops::BroadcastSrcNode(%28, %dglgraph)
  %z_s.1 : Tensor = aten::mm(%29, %fc_weight_msg)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %32 : Tensor = my_ops::BroadcastDstNode(%31, %dglgraph)
  %z_d.1 : Tensor = aten::mm(%32, %fc_weight_msg)
  %34 : Tensor[] = prim::ListConstruct(%z_s.1, %z_d.1)
  %z2.1 : Tensor = aten::cat(%34, %27)
  %a.1 : Tensor = aten::mm(%z2.1, %attn_weight_msg)
  %37 : Tensor = aten::leaky_relu_(%a.1, %23)
  %38 : str = prim::Constant[value="h"]()
  %39 : bool = prim::Constant[value=0]()
  %40 : str = prim::Constant[value="z"]()
  %41 : None = prim::Constant()
  %42 : str = prim::Constant[value="e"]()
  %43 : int = prim::Constant[value=1]()
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %43, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %z_s.1)
  %46 : int[] = prim::ListConstruct(%43)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%38, %47)
  return (%48)
After EliminateDeadCode =:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %23 : float = prim::Constant[value=0.01]()
  %26 : str = prim::Constant[value="h"]()
  %27 : int = prim::Constant[value=1]()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %29 : Tensor = my_ops::BroadcastSrcNode(%28, %dglgraph)
  %z_s.1 : Tensor = aten::mm(%29, %fc_weight_msg)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %32 : Tensor = my_ops::BroadcastDstNode(%31, %dglgraph)
  %z_d.1 : Tensor = aten::mm(%32, %fc_weight_msg)
  %34 : Tensor[] = prim::ListConstruct(%z_s.1, %z_d.1)
  %z2.1 : Tensor = aten::cat(%34, %27)
  %a.1 : Tensor = aten::mm(%z2.1, %attn_weight_msg)
  %37 : Tensor = aten::leaky_relu_(%a.1, %23)
  %38 : str = prim::Constant[value="h"]()
  %39 : bool = prim::Constant[value=0]()
  %41 : None = prim::Constant()
  %43 : int = prim::Constant[value=1]()
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %43, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %z_s.1)
  %46 : int[] = prim::ListConstruct(%43)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%38, %47)
  return (%48)
After ConstantPooling =:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %41 : None = prim::Constant()
  %39 : bool = prim::Constant[value=0]()
  %27 : int = prim::Constant[value=1]()
  %26 : str = prim::Constant[value="h"]()
  %23 : float = prim::Constant[value=0.01]()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %29 : Tensor = my_ops::BroadcastSrcNode(%28, %dglgraph)
  %z_s.1 : Tensor = aten::mm(%29, %fc_weight_msg)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %32 : Tensor = my_ops::BroadcastDstNode(%31, %dglgraph)
  %z_d.1 : Tensor = aten::mm(%32, %fc_weight_msg)
  %34 : Tensor[] = prim::ListConstruct(%z_s.1, %z_d.1)
  %z2.1 : Tensor = aten::cat(%34, %27)
  %a.1 : Tensor = aten::mm(%z2.1, %attn_weight_msg)
  %37 : Tensor = aten::leaky_relu_(%a.1, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %z_s.1)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
After EliminateCommonSubexpression =:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %41 : None = prim::Constant()
  %39 : bool = prim::Constant[value=0]()
  %27 : int = prim::Constant[value=1]()
  %26 : str = prim::Constant[value="h"]()
  %23 : float = prim::Constant[value=0.01]()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %29 : Tensor = my_ops::BroadcastSrcNode(%28, %dglgraph)
  %z_s.1 : Tensor = aten::mm(%29, %fc_weight_msg)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %32 : Tensor = my_ops::BroadcastDstNode(%31, %dglgraph)
  %z_d.1 : Tensor = aten::mm(%32, %fc_weight_msg)
  %34 : Tensor[] = prim::ListConstruct(%z_s.1, %z_d.1)
  %z2.1 : Tensor = aten::cat(%34, %27)
  %a.1 : Tensor = aten::mm(%z2.1, %attn_weight_msg)
  %37 : Tensor = aten::leaky_relu_(%a.1, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %z_s.1)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
After dedup =:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %41 : None = prim::Constant()
  %39 : bool = prim::Constant[value=0]()
  %27 : int = prim::Constant[value=1]()
  %26 : str = prim::Constant[value="h"]()
  %23 : float = prim::Constant[value=0.01]()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %29 : Tensor = my_ops::BroadcastSrcNode(%28, %dglgraph)
  %z_s.1 : Tensor = aten::mm(%29, %fc_weight_msg)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %32 : Tensor = my_ops::BroadcastDstNode(%31, %dglgraph)
  %z_d.1 : Tensor = aten::mm(%32, %fc_weight_msg)
  %34 : Tensor[] = prim::ListConstruct(%z_s.1, %z_d.1)
  %z2.1 : Tensor = aten::cat(%34, %27)
  %a.1 : Tensor = aten::mm(%z2.1, %attn_weight_msg)
  %37 : Tensor = aten::leaky_relu_(%a.1, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %z_s.1)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
Before Optimization
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %41 : None = prim::Constant()
  %39 : bool = prim::Constant[value=0]()
  %27 : int = prim::Constant[value=1]()
  %26 : str = prim::Constant[value="h"]()
  %23 : float = prim::Constant[value=0.01]()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %29 : Tensor = my_ops::BroadcastSrcNode(%28, %dglgraph)
  %z_s.1 : Tensor = aten::mm(%29, %fc_weight_msg)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %32 : Tensor = my_ops::BroadcastDstNode(%31, %dglgraph)
  %z_d.1 : Tensor = aten::mm(%32, %fc_weight_msg)
  %34 : Tensor[] = prim::ListConstruct(%z_s.1, %z_d.1)
  %z2.1 : Tensor = aten::cat(%34, %27)
  %a.1 : Tensor = aten::mm(%z2.1, %attn_weight_msg)
  %37 : Tensor = aten::leaky_relu_(%a.1, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %z_s.1)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)

Before Optimization Code
def mpdfg_func(dglgraph: __torch__.torch.classes.my_classes.DGLGraph,
    ndata: Dict[str, Tensor],
    edata: Dict[str, Tensor],
    ntypedata: Dict[str, Tensor],
    etypedata: Dict[str, Tensor],
    fc_weight_msg: Tensor,
    attn_weight_msg: Tensor) -> Dict[str, Tensor]:
  _0 = ops.my_ops.BroadcastSrcNode(ndata["h"], dglgraph)
  z_s = torch.mm(_0, fc_weight_msg)
  _1 = ops.my_ops.BroadcastDstNode(ndata["h"], dglgraph)
  z_d = torch.mm(_1, fc_weight_msg)
  z2 = torch.cat([z_s, z_d], 1)
  a = torch.mm(z2, attn_weight_msg)
  _2 = ops.my_ops.SegmentSoftmax(torch.leaky_relu_(a, 0.01), 1, None, dglgraph)
  _3 = ops.my_ops.SpMMEdge(torch.mul(_2, z_s), [1], False, None, dglgraph)
  return {"h": _3}

After Split
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %41 : None = prim::Constant()
  %39 : bool = prim::Constant[value=0]()
  %27 : int = prim::Constant[value=1]()
  %26 : str = prim::Constant[value="h"]()
  %23 : float = prim::Constant[value=0.01]()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %29 : Tensor = my_ops::BroadcastSrcNode(%28, %dglgraph)
  %z_s.1 : Tensor = aten::mm(%29, %fc_weight_msg)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %32 : Tensor = my_ops::BroadcastDstNode(%31, %dglgraph)
  %z_d.1 : Tensor = aten::mm(%32, %fc_weight_msg)
  %49 : int = prim::Constant[value=0]()
  %50 : int = prim::Constant[value=1]()
  %51 : int = my_ops::get_feat_dim(%z_s.1)
  %52 : int = my_ops::get_feat_dim(%z_d.1)
  %53 : int[] = prim::ListConstruct(%51, %52)
  %54 : Tensor[] = aten::split(%attn_weight_msg, %53, %49)
  %55 : Tensor, %56 : Tensor = prim::ListUnpack(%54)
  %57 : Tensor = aten::mm(%z_s.1, %55)
  %58 : Tensor = aten::mm(%z_d.1, %56)
  %59 : Tensor = aten::add(%57, %58, %50)
  %37 : Tensor = aten::leaky_relu_(%59, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %z_s.1)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
After EliminateDeadCode =:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %41 : None = prim::Constant()
  %39 : bool = prim::Constant[value=0]()
  %27 : int = prim::Constant[value=1]()
  %26 : str = prim::Constant[value="h"]()
  %23 : float = prim::Constant[value=0.01]()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %29 : Tensor = my_ops::BroadcastSrcNode(%28, %dglgraph)
  %z_s.1 : Tensor = aten::mm(%29, %fc_weight_msg)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %32 : Tensor = my_ops::BroadcastDstNode(%31, %dglgraph)
  %z_d.1 : Tensor = aten::mm(%32, %fc_weight_msg)
  %49 : int = prim::Constant[value=0]()
  %50 : int = prim::Constant[value=1]()
  %51 : int = my_ops::get_feat_dim(%z_s.1)
  %52 : int = my_ops::get_feat_dim(%z_d.1)
  %53 : int[] = prim::ListConstruct(%51, %52)
  %54 : Tensor[] = aten::split(%attn_weight_msg, %53, %49)
  %55 : Tensor, %56 : Tensor = prim::ListUnpack(%54)
  %57 : Tensor = aten::mm(%z_s.1, %55)
  %58 : Tensor = aten::mm(%z_d.1, %56)
  %59 : Tensor = aten::add(%57, %58, %50)
  %37 : Tensor = aten::leaky_relu_(%59, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %z_s.1)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
After ConstantPooling =:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %49 : int = prim::Constant[value=0]()
  %23 : float = prim::Constant[value=0.01]()
  %26 : str = prim::Constant[value="h"]()
  %27 : int = prim::Constant[value=1]()
  %39 : bool = prim::Constant[value=0]()
  %41 : None = prim::Constant()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %29 : Tensor = my_ops::BroadcastSrcNode(%28, %dglgraph)
  %z_s.1 : Tensor = aten::mm(%29, %fc_weight_msg)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %32 : Tensor = my_ops::BroadcastDstNode(%31, %dglgraph)
  %z_d.1 : Tensor = aten::mm(%32, %fc_weight_msg)
  %51 : int = my_ops::get_feat_dim(%z_s.1)
  %52 : int = my_ops::get_feat_dim(%z_d.1)
  %53 : int[] = prim::ListConstruct(%51, %52)
  %54 : Tensor[] = aten::split(%attn_weight_msg, %53, %49)
  %55 : Tensor, %56 : Tensor = prim::ListUnpack(%54)
  %57 : Tensor = aten::mm(%z_s.1, %55)
  %58 : Tensor = aten::mm(%z_d.1, %56)
  %59 : Tensor = aten::add(%57, %58, %27)
  %37 : Tensor = aten::leaky_relu_(%59, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %z_s.1)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
After EliminateCommonSubexpression =:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %49 : int = prim::Constant[value=0]()
  %23 : float = prim::Constant[value=0.01]()
  %26 : str = prim::Constant[value="h"]()
  %27 : int = prim::Constant[value=1]()
  %39 : bool = prim::Constant[value=0]()
  %41 : None = prim::Constant()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %29 : Tensor = my_ops::BroadcastSrcNode(%28, %dglgraph)
  %z_s.1 : Tensor = aten::mm(%29, %fc_weight_msg)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %32 : Tensor = my_ops::BroadcastDstNode(%31, %dglgraph)
  %z_d.1 : Tensor = aten::mm(%32, %fc_weight_msg)
  %51 : int = my_ops::get_feat_dim(%z_s.1)
  %52 : int = my_ops::get_feat_dim(%z_d.1)
  %53 : int[] = prim::ListConstruct(%51, %52)
  %54 : Tensor[] = aten::split(%attn_weight_msg, %53, %49)
  %55 : Tensor, %56 : Tensor = prim::ListUnpack(%54)
  %57 : Tensor = aten::mm(%z_s.1, %55)
  %58 : Tensor = aten::mm(%z_d.1, %56)
  %59 : Tensor = aten::add(%57, %58, %27)
  %37 : Tensor = aten::leaky_relu_(%59, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %z_s.1)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
After dedup =:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %49 : int = prim::Constant[value=0]()
  %23 : float = prim::Constant[value=0.01]()
  %26 : str = prim::Constant[value="h"]()
  %27 : int = prim::Constant[value=1]()
  %39 : bool = prim::Constant[value=0]()
  %41 : None = prim::Constant()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %29 : Tensor = my_ops::BroadcastSrcNode(%28, %dglgraph)
  %z_s.1 : Tensor = aten::mm(%29, %fc_weight_msg)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %32 : Tensor = my_ops::BroadcastDstNode(%31, %dglgraph)
  %z_d.1 : Tensor = aten::mm(%32, %fc_weight_msg)
  %51 : int = my_ops::get_feat_dim(%z_s.1)
  %52 : int = my_ops::get_feat_dim(%z_d.1)
  %53 : int[] = prim::ListConstruct(%51, %52)
  %54 : Tensor[] = aten::split(%attn_weight_msg, %53, %49)
  %55 : Tensor, %56 : Tensor = prim::ListUnpack(%54)
  %57 : Tensor = aten::mm(%z_s.1, %55)
  %58 : Tensor = aten::mm(%z_d.1, %56)
  %59 : Tensor = aten::add(%57, %58, %27)
  %37 : Tensor = aten::leaky_relu_(%59, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %z_s.1)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
After Reorder
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %49 : int = prim::Constant[value=0]()
  %23 : float = prim::Constant[value=0.01]()
  %26 : str = prim::Constant[value="h"]()
  %27 : int = prim::Constant[value=1]()
  %39 : bool = prim::Constant[value=0]()
  %41 : None = prim::Constant()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %60 : Tensor = my_ops::BroadcastSrcNode(%28, %dglgraph)
  %61 : Tensor = aten::mm(%28, %fc_weight_msg)
  %63 : int = my_ops::get_feat_dim(%61)
  %64 : Tensor = my_ops::BroadcastSrcNode(%61, %dglgraph)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %65 : Tensor = my_ops::BroadcastDstNode(%31, %dglgraph)
  %66 : Tensor = aten::mm(%31, %fc_weight_msg)
  %68 : int = my_ops::get_feat_dim(%66)
  %69 : Tensor = my_ops::BroadcastDstNode(%66, %dglgraph)
  %53 : int[] = prim::ListConstruct(%63, %68)
  %54 : Tensor[] = aten::split(%attn_weight_msg, %53, %49)
  %55 : Tensor, %56 : Tensor = prim::ListUnpack(%54)
  %57 : Tensor = aten::mm(%64, %55)
  %58 : Tensor = aten::mm(%69, %56)
  %59 : Tensor = aten::add(%57, %58, %27)
  %37 : Tensor = aten::leaky_relu_(%59, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %64)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
After EliminateDeadCode =:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %49 : int = prim::Constant[value=0]()
  %23 : float = prim::Constant[value=0.01]()
  %26 : str = prim::Constant[value="h"]()
  %27 : int = prim::Constant[value=1]()
  %39 : bool = prim::Constant[value=0]()
  %41 : None = prim::Constant()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %60 : Tensor = my_ops::BroadcastSrcNode(%28, %dglgraph)
  %61 : Tensor = aten::mm(%28, %fc_weight_msg)
  %63 : int = my_ops::get_feat_dim(%61)
  %64 : Tensor = my_ops::BroadcastSrcNode(%61, %dglgraph)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %65 : Tensor = my_ops::BroadcastDstNode(%31, %dglgraph)
  %66 : Tensor = aten::mm(%31, %fc_weight_msg)
  %68 : int = my_ops::get_feat_dim(%66)
  %69 : Tensor = my_ops::BroadcastDstNode(%66, %dglgraph)
  %53 : int[] = prim::ListConstruct(%63, %68)
  %54 : Tensor[] = aten::split(%attn_weight_msg, %53, %49)
  %55 : Tensor, %56 : Tensor = prim::ListUnpack(%54)
  %57 : Tensor = aten::mm(%64, %55)
  %58 : Tensor = aten::mm(%69, %56)
  %59 : Tensor = aten::add(%57, %58, %27)
  %37 : Tensor = aten::leaky_relu_(%59, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %64)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
After ConstantPooling =:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %41 : None = prim::Constant()
  %39 : bool = prim::Constant[value=0]()
  %27 : int = prim::Constant[value=1]()
  %26 : str = prim::Constant[value="h"]()
  %23 : float = prim::Constant[value=0.01]()
  %49 : int = prim::Constant[value=0]()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %60 : Tensor = my_ops::BroadcastSrcNode(%28, %dglgraph)
  %61 : Tensor = aten::mm(%28, %fc_weight_msg)
  %63 : int = my_ops::get_feat_dim(%61)
  %64 : Tensor = my_ops::BroadcastSrcNode(%61, %dglgraph)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %65 : Tensor = my_ops::BroadcastDstNode(%31, %dglgraph)
  %66 : Tensor = aten::mm(%31, %fc_weight_msg)
  %68 : int = my_ops::get_feat_dim(%66)
  %69 : Tensor = my_ops::BroadcastDstNode(%66, %dglgraph)
  %53 : int[] = prim::ListConstruct(%63, %68)
  %54 : Tensor[] = aten::split(%attn_weight_msg, %53, %49)
  %55 : Tensor, %56 : Tensor = prim::ListUnpack(%54)
  %57 : Tensor = aten::mm(%64, %55)
  %58 : Tensor = aten::mm(%69, %56)
  %59 : Tensor = aten::add(%57, %58, %27)
  %37 : Tensor = aten::leaky_relu_(%59, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %64)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
After EliminateCommonSubexpression =:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %41 : None = prim::Constant()
  %39 : bool = prim::Constant[value=0]()
  %27 : int = prim::Constant[value=1]()
  %26 : str = prim::Constant[value="h"]()
  %23 : float = prim::Constant[value=0.01]()
  %49 : int = prim::Constant[value=0]()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %60 : Tensor = my_ops::BroadcastSrcNode(%28, %dglgraph)
  %61 : Tensor = aten::mm(%28, %fc_weight_msg)
  %63 : int = my_ops::get_feat_dim(%61)
  %64 : Tensor = my_ops::BroadcastSrcNode(%61, %dglgraph)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %65 : Tensor = my_ops::BroadcastDstNode(%31, %dglgraph)
  %66 : Tensor = aten::mm(%31, %fc_weight_msg)
  %68 : int = my_ops::get_feat_dim(%66)
  %69 : Tensor = my_ops::BroadcastDstNode(%66, %dglgraph)
  %53 : int[] = prim::ListConstruct(%63, %68)
  %54 : Tensor[] = aten::split(%attn_weight_msg, %53, %49)
  %55 : Tensor, %56 : Tensor = prim::ListUnpack(%54)
  %57 : Tensor = aten::mm(%64, %55)
  %58 : Tensor = aten::mm(%69, %56)
  %59 : Tensor = aten::add(%57, %58, %27)
  %37 : Tensor = aten::leaky_relu_(%59, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %64)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
After dedup =:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %41 : None = prim::Constant()
  %39 : bool = prim::Constant[value=0]()
  %27 : int = prim::Constant[value=1]()
  %26 : str = prim::Constant[value="h"]()
  %23 : float = prim::Constant[value=0.01]()
  %49 : int = prim::Constant[value=0]()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %61 : Tensor = aten::mm(%28, %fc_weight_msg)
  %63 : int = my_ops::get_feat_dim(%61)
  %64 : Tensor = my_ops::BroadcastSrcNode(%61, %dglgraph)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %66 : Tensor = aten::mm(%31, %fc_weight_msg)
  %68 : int = my_ops::get_feat_dim(%66)
  %69 : Tensor = my_ops::BroadcastDstNode(%66, %dglgraph)
  %53 : int[] = prim::ListConstruct(%63, %68)
  %54 : Tensor[] = aten::split(%attn_weight_msg, %53, %49)
  %55 : Tensor, %56 : Tensor = prim::ListUnpack(%54)
  %57 : Tensor = aten::mm(%64, %55)
  %58 : Tensor = aten::mm(%69, %56)
  %59 : Tensor = aten::add(%57, %58, %27)
  %37 : Tensor = aten::leaky_relu_(%59, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %64)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
After Reorder
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %41 : None = prim::Constant()
  %39 : bool = prim::Constant[value=0]()
  %27 : int = prim::Constant[value=1]()
  %26 : str = prim::Constant[value="h"]()
  %23 : float = prim::Constant[value=0.01]()
  %49 : int = prim::Constant[value=0]()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %61 : Tensor = aten::mm(%28, %fc_weight_msg)
  %63 : int = my_ops::get_feat_dim(%61)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %66 : Tensor = aten::mm(%31, %fc_weight_msg)
  %68 : int = my_ops::get_feat_dim(%66)
  %53 : int[] = prim::ListConstruct(%63, %68)
  %54 : Tensor[] = aten::split(%attn_weight_msg, %53, %49)
  %55 : Tensor, %56 : Tensor = prim::ListUnpack(%54)
  %73 : Tensor = my_ops::BroadcastDstNode(%66, %dglgraph)
  %74 : Tensor = aten::mm(%66, %56)
  %75 : Tensor = my_ops::BroadcastDstNode(%74, %dglgraph)
  %70 : Tensor = my_ops::BroadcastSrcNode(%61, %dglgraph)
  %71 : Tensor = aten::mm(%61, %55)
  %72 : Tensor = my_ops::BroadcastSrcNode(%71, %dglgraph)
  %59 : Tensor = aten::add(%72, %75, %27)
  %37 : Tensor = aten::leaky_relu_(%59, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %70)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
After EliminateDeadCode =:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %41 : None = prim::Constant()
  %39 : bool = prim::Constant[value=0]()
  %27 : int = prim::Constant[value=1]()
  %26 : str = prim::Constant[value="h"]()
  %23 : float = prim::Constant[value=0.01]()
  %49 : int = prim::Constant[value=0]()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %61 : Tensor = aten::mm(%28, %fc_weight_msg)
  %63 : int = my_ops::get_feat_dim(%61)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %66 : Tensor = aten::mm(%31, %fc_weight_msg)
  %68 : int = my_ops::get_feat_dim(%66)
  %53 : int[] = prim::ListConstruct(%63, %68)
  %54 : Tensor[] = aten::split(%attn_weight_msg, %53, %49)
  %55 : Tensor, %56 : Tensor = prim::ListUnpack(%54)
  %73 : Tensor = my_ops::BroadcastDstNode(%66, %dglgraph)
  %74 : Tensor = aten::mm(%66, %56)
  %75 : Tensor = my_ops::BroadcastDstNode(%74, %dglgraph)
  %70 : Tensor = my_ops::BroadcastSrcNode(%61, %dglgraph)
  %71 : Tensor = aten::mm(%61, %55)
  %72 : Tensor = my_ops::BroadcastSrcNode(%71, %dglgraph)
  %59 : Tensor = aten::add(%72, %75, %27)
  %37 : Tensor = aten::leaky_relu_(%59, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %70)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
After ConstantPooling =:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %49 : int = prim::Constant[value=0]()
  %23 : float = prim::Constant[value=0.01]()
  %26 : str = prim::Constant[value="h"]()
  %27 : int = prim::Constant[value=1]()
  %39 : bool = prim::Constant[value=0]()
  %41 : None = prim::Constant()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %61 : Tensor = aten::mm(%28, %fc_weight_msg)
  %63 : int = my_ops::get_feat_dim(%61)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %66 : Tensor = aten::mm(%31, %fc_weight_msg)
  %68 : int = my_ops::get_feat_dim(%66)
  %53 : int[] = prim::ListConstruct(%63, %68)
  %54 : Tensor[] = aten::split(%attn_weight_msg, %53, %49)
  %55 : Tensor, %56 : Tensor = prim::ListUnpack(%54)
  %73 : Tensor = my_ops::BroadcastDstNode(%66, %dglgraph)
  %74 : Tensor = aten::mm(%66, %56)
  %75 : Tensor = my_ops::BroadcastDstNode(%74, %dglgraph)
  %70 : Tensor = my_ops::BroadcastSrcNode(%61, %dglgraph)
  %71 : Tensor = aten::mm(%61, %55)
  %72 : Tensor = my_ops::BroadcastSrcNode(%71, %dglgraph)
  %59 : Tensor = aten::add(%72, %75, %27)
  %37 : Tensor = aten::leaky_relu_(%59, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %70)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
After EliminateCommonSubexpression =:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %49 : int = prim::Constant[value=0]()
  %23 : float = prim::Constant[value=0.01]()
  %26 : str = prim::Constant[value="h"]()
  %27 : int = prim::Constant[value=1]()
  %39 : bool = prim::Constant[value=0]()
  %41 : None = prim::Constant()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %61 : Tensor = aten::mm(%28, %fc_weight_msg)
  %63 : int = my_ops::get_feat_dim(%61)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %66 : Tensor = aten::mm(%31, %fc_weight_msg)
  %68 : int = my_ops::get_feat_dim(%66)
  %53 : int[] = prim::ListConstruct(%63, %68)
  %54 : Tensor[] = aten::split(%attn_weight_msg, %53, %49)
  %55 : Tensor, %56 : Tensor = prim::ListUnpack(%54)
  %73 : Tensor = my_ops::BroadcastDstNode(%66, %dglgraph)
  %74 : Tensor = aten::mm(%66, %56)
  %75 : Tensor = my_ops::BroadcastDstNode(%74, %dglgraph)
  %70 : Tensor = my_ops::BroadcastSrcNode(%61, %dglgraph)
  %71 : Tensor = aten::mm(%61, %55)
  %72 : Tensor = my_ops::BroadcastSrcNode(%71, %dglgraph)
  %59 : Tensor = aten::add(%72, %75, %27)
  %37 : Tensor = aten::leaky_relu_(%59, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %70)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
After dedup =:
graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %49 : int = prim::Constant[value=0]()
  %23 : float = prim::Constant[value=0.01]()
  %26 : str = prim::Constant[value="h"]()
  %27 : int = prim::Constant[value=1]()
  %39 : bool = prim::Constant[value=0]()
  %41 : None = prim::Constant()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %61 : Tensor = aten::mm(%28, %fc_weight_msg)
  %63 : int = my_ops::get_feat_dim(%61)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %66 : Tensor = aten::mm(%31, %fc_weight_msg)
  %68 : int = my_ops::get_feat_dim(%66)
  %53 : int[] = prim::ListConstruct(%63, %68)
  %54 : Tensor[] = aten::split(%attn_weight_msg, %53, %49)
  %55 : Tensor, %56 : Tensor = prim::ListUnpack(%54)
  %74 : Tensor = aten::mm(%66, %56)
  %75 : Tensor = my_ops::BroadcastDstNode(%74, %dglgraph)
  %70 : Tensor = my_ops::BroadcastSrcNode(%61, %dglgraph)
  %71 : Tensor = aten::mm(%61, %55)
  %72 : Tensor = my_ops::BroadcastSrcNode(%71, %dglgraph)
  %59 : Tensor = aten::add(%72, %75, %27)
  %37 : Tensor = aten::leaky_relu_(%59, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %70)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, %46, %39, %41, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)
Afrer Optimization
MPDFG:
 graph(%dglgraph : __torch__.torch.classes.my_classes.DGLGraph,
      %ndata : Dict(str, Tensor),
      %edata : Dict(str, Tensor),
      %ntypedata : Dict(str, Tensor),
      %etypedata : Dict(str, Tensor),
      %fc_weight_msg : Tensor,
      %attn_weight_msg : Tensor):
  %49 : int = prim::Constant[value=0]()
  %23 : float = prim::Constant[value=0.01]()
  %26 : str = prim::Constant[value="h"]()
  %27 : int = prim::Constant[value=1]()
  %39 : bool = prim::Constant[value=0]()
  %41 : None = prim::Constant()
  %28 : Tensor = aten::__getitem__(%ndata, %26)
  %61 : Tensor = aten::mm(%28, %fc_weight_msg)
  %63 : int = my_ops::get_feat_dim(%61)
  %31 : Tensor = aten::__getitem__(%ndata, %26)
  %66 : Tensor = aten::mm(%31, %fc_weight_msg)
  %68 : int = my_ops::get_feat_dim(%66)
  %53 : int[] = prim::ListConstruct(%63, %68)
  %54 : Tensor[] = aten::split(%attn_weight_msg, %53, %49)
  %55 : Tensor, %56 : Tensor = prim::ListUnpack(%54)
  %74 : Tensor = aten::mm(%66, %56)
  %75 : Tensor = my_ops::BroadcastDstNode(%74, %dglgraph)
  %70 : Tensor = my_ops::BroadcastSrcNode(%61, %dglgraph)
  %71 : Tensor = aten::mm(%61, %55)
  %72 : Tensor = my_ops::BroadcastSrcNode(%71, %dglgraph)
  %59 : Tensor = aten::add(%72, %75, %27)
  %37 : Tensor = aten::leaky_relu_(%59, %23)
  %44 : Tensor = my_ops::SegmentSoftmax(%37, %27, %41, %dglgraph)
  %45 : Tensor = aten::mul(%44, %70)
  %46 : int[] = prim::ListConstruct(%27)
  %47 : Tensor = my_ops::SpMMEdge(%45, 1, 0, None, %dglgraph)
  %48 : Dict(str, Tensor) = prim::DictConstruct(%26, %47)
  return (%48)

TorchScript Code:
def mpdfg_func(dglgraph: __torch__.torch.classes.my_classes.DGLGraph,
    ndata: Dict[str, Tensor],
    edata: Dict[str, Tensor],
    ntypedata: Dict[str, Tensor],
    etypedata: Dict[str, Tensor],
    fc_weight_msg: Tensor,
    attn_weight_msg: Tensor) -> Dict[str, Tensor]:
  _0 = torch.mm(ndata["h"], fc_weight_msg)
  _1 = ops.my_ops.get_feat_dim(_0)
  _2 = torch.mm(ndata["h"], fc_weight_msg)
  _3 = torch.split(attn_weight_msg, [_1, ops.my_ops.get_feat_dim(_2)], 0)
  _4, _5, = _3
  _6 = ops.my_ops.BroadcastDstNode(torch.mm(_2, _5), dglgraph)
  _7 = ops.my_ops.BroadcastSrcNode(_0, dglgraph)
  _8 = ops.my_ops.BroadcastSrcNode(torch.mm(_0, _4), dglgraph)
  _9 = torch.leaky_relu_(torch.add(_8, _6, alpha=1), 0.01)
  _10 = ops.my_ops.SegmentSoftmax(_9, 1, None, dglgraph)
  _11 = ops.my_ops.SpMMEdge(torch.mul(_10, _7), [1], False, None, dglgraph)
  return {"h": _11}