#include "builder.h"
#include <iostream>
#include <vector>
static bool UPDATE_UDF = false;

namespace graphiler {
using namespace torch::jit;

// mapping of unique id to Value
static std::unordered_map<size_t, Value *> VALUE_MAP{};
static std::unordered_map<size_t, Movement> VALUE_TO_BROADCAST{};

static std::unordered_set<size_t> VALUE_FOR_MAIL;
// string seems to be the only valid key
static std::unordered_map<std::string, Value *> MAIL_BOX;

// store the constant value for Values
static std::unordered_map<size_t, std::vector<IValue>> CONSTANT_MAP{};

// Todo: more reduce and norm type operators in a more organized
const static std::unordered_map<std::string, Movement> MOVEMENT_MAP{
    {"my_ops::SegmentSoftmax", Movement::Norm},
    {"my_ops::SpMMEdge", Movement::Reduce}};
std::string
operator_lowering(const std::string &n_kind,
                  const std::vector<Value *> &new_inputs,
                  const std::vector<Residency> &new_inputs_residency) {
  if ((n_kind == std::string("aten::softmax")) &&
      (new_inputs[1]->node()->i(attr::value) == 1) &&
      (new_inputs_residency[0] == Residency::Edge)) {
    return "my_ops::SegmentSoftmax";
  }
  if ((n_kind == std::string("aten::sum")) &&
      (CONSTANT_MAP.find(new_inputs[1]->unique()) != CONSTANT_MAP.end()) &&
      (new_inputs_residency[0] == Residency::Edge)) {
    auto dims = CONSTANT_MAP[new_inputs[1]->unique()];
    assert(dims.size() == 1);
    if (dims[0].toIntList()[0] == 1) {
      return "my_ops::SpMMEdge";
    }
  }
  return "";
}

void print_graph(Block *in_block) {
  int node_id  = 0;
  for (auto n : in_block->nodes()) {
    std::string n_kind = n->kind().toQualString();
    std::cout << n_kind + std::to_string(node_id) << "[shape=circle]" << std::endl;
    node_id++;
  }
}

void parse_stage(std::shared_ptr<MPDFGAnnotation> &mpdfg,
                 c10::ArrayRef<Value *> mpdfg_params, Block *in_block,
                 Stage stage) {
  // if(stage != Stage::Creation)
  // std::cout << "Begin Parse" << std::endl;
  auto dglgraph = mpdfg_params[0];
  auto mpdfg_block = mpdfg->DFG->block();
  auto in_final_node = in_block->nodes().end()->input()->node();

  
  auto mpdfg_final_node = mpdfg_block->nodes().end()->input()->node();

  // if(stage != Stage::Creation)
  // std::cout << "mpdfg_block: ";
  
  // if(stage != Stage::Creation)
  // mpdfg_block->owningGraph()->print(std::cout);

  // if(stage != Stage::Creation)
  // std::cout << "mpdfg_block final node: ";
  // if(stage != Stage::Creation)
  // mpdfg_final_node->dump();
  print_graph(in_block);
  for (auto n : in_block->nodes()) {
    // if(stage != Stage::Creation)
    // std::cout << "\nNode: ";
    // if(stage != Stage::Creation)
    // n->dump();
    
    std::string n_kind = n->kind().toQualString();
    // if(stage != Stage::Creation)
    // std::cout << "Node Kind = " << n_kind << std::endl;

    // Todo: modularize different parts of the function

    // find the root varibles through DGL APIs
    // Todo: GetAttr or CallMethod?
    if ((n_kind == std::string("prim::GetAttr")) &&
        (n->hasAttributeS("name"))) {
      auto output_id = n->output()->unique();
      std::string attr_name = n->s(attr::name);

      if (attr_name == "_src_data") {
        VALUE_MAP[output_id] = mpdfg_params[1];
        VALUE_TO_BROADCAST[output_id] = Movement::BroadcastSrc;
        continue;
      } else if (attr_name == "_dst_data") {
        VALUE_MAP[output_id] = mpdfg_params[1];
        VALUE_TO_BROADCAST[output_id] = Movement::BroadcastDst;
        continue;
      } else if (attr_name == "_edge_data") {
        VALUE_MAP[output_id] = mpdfg_params[2];
        continue;
      } else if (attr_name == "_srctype_data") {
        VALUE_MAP[output_id] = mpdfg_params[3];
        VALUE_TO_BROADCAST[output_id] = Movement::BroadcastSrc;
        continue;
      } else if (attr_name == "_dsttype_data") {
        VALUE_MAP[output_id] = mpdfg_params[3];
        VALUE_TO_BROADCAST[output_id] = Movement::BroadcastDst;
        continue;
      } else if (attr_name == "_edgetype_data") {
        VALUE_MAP[output_id] = mpdfg_params[4];
        VALUE_TO_BROADCAST[output_id] = Movement::Broadcast;
        continue;
      } else if (attr_name == "_msgs") {
        VALUE_FOR_MAIL.insert(output_id);
        VALUE_MAP[output_id] = mpdfg_params[2];
        continue;
      } else if (attr_name == "_node_data") {
        VALUE_MAP[output_id] = mpdfg_params[1];
        continue;
      } else if (attr_name == "_nodetype_data") {
        VALUE_MAP[output_id] = mpdfg_params[3];
        VALUE_TO_BROADCAST[output_id] = Movement::Broadcast;
        continue;
      }
    }

    // map inputs in original functions to MP-DFG
    std::vector<Value *> new_inputs;
    std::vector<Residency> new_inputs_residency;

    // if(stage != Stage::Creation)
    // std::cout << "Old Input\n";
    for (auto i : n->inputs()) {
      // if(stage != Stage::Creation)
      // std::cout << "Input: " << i->debugName() << std::endl;
      assert(VALUE_MAP.find(i->unique()) != VALUE_MAP.end());
      new_inputs.push_back(VALUE_MAP[i->unique()]);
      new_inputs_residency.push_back(
          mpdfg->data_residency[new_inputs.back()->unique()]);
    }
    // if(stage != Stage::Creation)
    // std::cout << "New Input\n";
    // for(auto i : new_inputs) {
    //   if(stage != Stage::Creation)
    //   std::cout << "Input: " << i->debugName() << std::endl;
    // }

    if ((stage == Stage::Creation) && (n == in_final_node)) {
      // if(stage != Stage::Creation)
      // std::cout << "Node is Msg Final Node" << std::endl;
      // construct mailbox in the message creation stage
      assert(new_inputs.size() % 2 == 0);
      assert(n_kind == std::string("prim::DictConstruct"));
      for (size_t o = 0; o < new_inputs.size(); o += 2) {
        MAIL_BOX[new_inputs[o]->node()->s(attr::value)] = new_inputs[o + 1];
        // auto key = toIValue(new_inputs[o]);
        // assert(key);
        // MAIL_BOX[key.value()] = new_inputs[o + 1];
      }
      // if(stage != Stage::Creation)
      // std::cout << "Mail Box\n";
      // for(auto m : MAIL_BOX) {
      //   if(stage != Stage::Creation)
      //   std::cout << m.first << ", " << m.second->debugName() << std::endl;
      // }

      continue;
    } else if ((stage == Stage::Aggregation) &&
               (n_kind == std::string("aten::__getitem__")) &&
               (VALUE_FOR_MAIL.count(n->inputs()[0]->unique()) != 0)) {
      // extract data from the mailbox
      assert(MAIL_BOX.find(new_inputs[1]->node()->s(attr::value)) !=
             MAIL_BOX.end());
      VALUE_MAP[n->output()->unique()] =
          MAIL_BOX[new_inputs[1]->node()->s(attr::value)];
      // auto key = toIValue(new_inputs[1]);
      // assert(key);
      // assert(MAIL_BOX.find(key) != MAIL_BOX.end());
      // VALUE_MAP[n->output()->unique()] = MAIL_BOX[key];
      continue;
    }

    // rules to identify reduce and norm type operators in the Aggregation stage
    if (stage == Stage::Aggregation) {
      auto new_kind_info = MOVEMENT_MAP.find(
          operator_lowering(n_kind, new_inputs, new_inputs_residency));
      if (new_kind_info != MOVEMENT_MAP.end()) {
        auto new_kind = NodeKind::fromQualString(new_kind_info->first);
        new_inputs.push_back(dglgraph);
        auto new_node = mpdfg->DFG->create(new_kind, new_inputs, 1);
        new_node->insertBefore(mpdfg_final_node);
        VALUE_MAP[n->output()->unique()] = new_node->output();

        mpdfg->data_movement[new_node] = new_kind_info->second;
        mpdfg->data_residency[new_node->output()->unique()] =
            (new_kind_info->second == Movement::Norm) ? Residency::Edge
                                                      : Residency::Node;
        

        // if(stage != Stage::Creation)
        // mpdfg->DFG->print(std::cout);

        continue;
      }
    }

    // copy nodes from original functions to MP-DFG
    auto new_node =
        mpdfg->DFG->create(n->kind(), new_inputs, n->outputs().size());
    // FTXJ, create a node with spec input and spec output number.
    new_node->copyAttributes(*n);
    new_node->insertBefore(mpdfg_final_node);

    // if(stage != Stage::Creation)
    // std::cout << "Old Outputs\n";
    for (size_t o = 0; o < n->outputs().size(); o++) {
      // if(stage != Stage::Creation)
      // std::cout << "Output: " << n->outputs()[o]->debugName() << std::endl;
      new_node->outputs()[o]->copyMetadata(n->outputs()[o]);
      VALUE_MAP[n->outputs()[o]->unique()] = new_node->outputs()[o];
    }

    // if(stage != Stage::Creation)
    // std::cout << "New Outputs\n";
    // for (size_t o = 0; o < new_node->outputs().size(); o++) {
    //   if(stage != Stage::Creation)
    //   std::cout << "Output: " << new_node->outputs()[o]->debugName() << std::endl;
    // }
    // default movement types of operators
    mpdfg->data_movement[new_node] = Movement::Dense;

    // operator specific semantics
    auto first_output_id = new_node->outputs()[0]->unique();
  

    if (n_kind == std::string("aten::__getitem__")) {
      auto v_id = n->inputs()[0]->unique();
      if (VALUE_TO_BROADCAST.find(v_id) != VALUE_TO_BROADCAST.end()) {
        // insert broadcast operator for data fetching on demand
        NodeKind new_kind;
        auto v_residency = mpdfg->data_residency[VALUE_MAP[v_id]->unique()];
        switch (VALUE_TO_BROADCAST[v_id]) {
        case Movement::BroadcastSrc:
          new_kind =
              (v_residency == Residency::Node)
                  ? NodeKind::fromQualString("my_ops::BroadcastSrcNode")
                  : NodeKind::fromQualString("my_ops::BroadcastSrcNodeType");
          break;
        case Movement::BroadcastDst:
          new_kind =
              (v_residency == Residency::Node)
                  ? NodeKind::fromQualString("my_ops::BroadcastDstNode")
                  : NodeKind::fromQualString("my_ops::BroadcastDstNodeType");
          break;
        case Movement::Broadcast:
          new_kind =
              (stage == Stage::Creation)
                  ? NodeKind::fromQualString("my_ops::BroadcastEdgeType")
                  : NodeKind::fromQualString("my_ops::BroadcastNodeType");
          break;
        default:
          assert(false && "unexpected data movement");
        }
        std::vector<Value *> broadcast_inputs = {new_node->output(), dglgraph};
        auto broadcast_node = mpdfg->DFG->create(new_kind, broadcast_inputs, 1);
        broadcast_node->insertAfter(new_node);
        VALUE_MAP[n->output()->unique()] = broadcast_node->output();
        // message creation stage is edge centric
        // while other two are node centric
        mpdfg->data_residency[broadcast_node->output()->unique()] =
            (stage == Stage::Creation) ? Residency::Edge : Residency::Node;
        mpdfg->data_movement[broadcast_node] = VALUE_TO_BROADCAST[v_id];
      } else {
        // ordinary __getitem__, no broadcast involved
        mpdfg->data_residency[first_output_id] = new_inputs_residency[0];
      }
    }
    // semantic checking for known operators
    else if (n_kind == std::string("aten::cat")) {
      if (new_inputs_residency[0] != Residency::Shared) {
        // if input0 is data associated with graph, e.g., [ndata0, ndata1]
        // for 2D matrix (N, f), the dim should only be 1
        assert(new_inputs[1]->node()->i(attr::value) == 1);
      }
      mpdfg->data_residency[first_output_id] = new_inputs_residency[0];
    } else if (n_kind == std::string("aten::softmax")) {
      if (new_inputs_residency[0] != Residency::Shared) {
        // not a valid operation in message passing
        assert(new_inputs[1]->node()->i(attr::value) != 0);
      }
      mpdfg->data_residency[first_output_id] = new_inputs_residency[0];
    } else if (n_kind == std::string("aten::sum")) {
      if (new_inputs_residency[0] != Residency::Shared) {
        if (CONSTANT_MAP.find(new_inputs[1]->unique()) != CONSTANT_MAP.end()) {
          auto dims = CONSTANT_MAP[new_inputs[1]->unique()][0].toIntList();
          // not a valid operation in message passing
          assert(dims.find(0) == dims.end());
        }
      }
      mpdfg->data_residency[first_output_id] = new_inputs_residency[0];
    }
    // Default semantic for Constant and other operators without inputs
    else if ((n_kind == std::string("prim::Constant")) ||
             (new_inputs_residency.size() == 0)) {
      mpdfg->data_residency[first_output_id] = Residency::Shared;
    }
    // Default semantic for dense operators
    else {
      if (std::equal(new_inputs_residency.begin() + 1,
                     new_inputs_residency.end(),
                     new_inputs_residency.begin())) {
        mpdfg->data_residency[first_output_id] = new_inputs_residency[0];
      } else {
        assert(new_inputs_residency.find(Residency::Shared) !=
               new_inputs_residency.end());
        Residency r_dominant = Residency::Shared;
        for (auto r : new_inputs_residency) {
          assert((r == Residency::Shared) || (r == r_dominant));
          if (r != Residency::Shared) {
            r_dominant = r;
          }
        }
        mpdfg->data_residency[first_output_id] = r_dominant;
      }
      // Todo: warning if operator not known
      // (n_kind == std::string("aten::mm")) ||
      //     (n_kind == std::string("prim::ListConstruct")) ||
      //     (n_kind == std::string("aten::relu")) ||
      //     (n_kind == std::string("aten::mul"))
    }

    // propagate constants
    auto optional_outputs = runNodeIfInputsAreConstant(n);
    if (optional_outputs) {
      CONSTANT_MAP[first_output_id] = optional_outputs.value();
    }

    if (UPDATE_UDF && (stage == Stage::Aggregation) && (n == in_final_node)) {
      // update node data at the end of message aggregation stage
      assert(new_inputs.size() % 2 == 0);
      assert(n_kind == std::string("prim::DictConstruct"));
      auto set_kind = NodeKind::fromQualString("aten::_set_item");
      for (size_t o = 0; o < new_inputs.size(); o += 2) {
        std::vector<Value *> set_inputs = {mpdfg_params[1], new_inputs[o],
                                           new_inputs[o + 1]};
        auto set_node = mpdfg->DFG->create(set_kind, set_inputs, 0);
        set_node->insertAfter(new_node);
      }
    } else if ((n == in_final_node) &&
               ((stage == Stage::Update) ||
                ((!UPDATE_UDF) && (stage == Stage::Aggregation)))) {
      // replace the final node to return
      mpdfg_final_node->replaceAllUsesWith(new_node);
      mpdfg_final_node->removeAllInputs();
      mpdfg_final_node->destroy();
    }
    // if(stage != Stage::Creation)
    // mpdfg->DFG->print(std::cout);
  }
}

void MPDFGBuilder(std::shared_ptr<MPDFGAnnotation> &mpdfg,
                std::shared_ptr<Graph> &msg_graph,
                std::shared_ptr<Graph> &reduce_graph,
                at::optional<std::shared_ptr<Graph>> update_graph) {

  Inline(*(mpdfg->DFG));
  std::cout << "Before Buildre MP-DFG:" << std::endl;
  mpdfg->DFG->print(std::cout);

  Inline(*msg_graph);
  std::cout << "Before Buildre Msg-DFG:" << std::endl;
  msg_graph->print(std::cout);
  
  Inline(*reduce_graph);
  std::cout << "Before Buildre Reduce-DFG:" << std::endl;
  reduce_graph->print(std::cout);

  // Todo: control flow
  auto mpdfg_block = mpdfg->DFG->block();
  auto msg_block = msg_graph->block();
  auto reduce_block = reduce_graph->block();
  Block *update_block;

  auto mpdfg_params = mpdfg_block->param_node()->outputs();
  auto msg_params = msg_block->param_node()->outputs();
  auto reduce_params = reduce_block->param_node()->outputs();

  size_t num_msg_params = msg_params.size();
  size_t num_reduce_params = reduce_params.size();
  
  // std::cout << "mpdfg_params:" << std::endl;
  // for(auto i = 0; i < mpdfg_params.size(); ++i) {
  //   std::cout << mpdfg_params[i]->unique() << ", " << mpdfg_params[i]->debugName() << std::endl;
  // }

  // std::cout << "msg_params:" << std::endl;
  // for(auto i = 0; i < msg_params.size(); ++i) {
  //   std::cout << msg_params[i]->unique() << ", " << msg_params[i]->debugName() << std::endl;
  // }
  
  // std::cout << "reduce_params:" << std::endl;
  // for(auto i = 0; i < reduce_params.size(); ++i) {
  //   std::cout << reduce_params[i]->unique() << ", " << reduce_params[i]->debugName() << std::endl;
  // }

  mpdfg->data_residency[mpdfg_params[1]->unique()] = Residency::Node;
  mpdfg->data_residency[mpdfg_params[2]->unique()] = Residency::Edge;
  mpdfg->data_residency[mpdfg_params[3]->unique()] = Residency::NodeType;
  mpdfg->data_residency[mpdfg_params[4]->unique()] = Residency::EdgeType;

  // map the parameters in the message function to MP-DFG
  // Todo: dedup
  size_t param_offset = 1;
  size_t param_pointer = 4;
  while (num_msg_params > param_offset) {
    VALUE_MAP[msg_params[param_offset]->unique()] =
        mpdfg_params[param_offset + param_pointer];
    mpdfg
        ->data_residency[mpdfg_params[param_offset + param_pointer]->unique()] =
        Residency::Shared;
    param_offset += 1;
  }
  param_offset = 1;
  param_pointer = num_msg_params + 3;
  while (num_reduce_params > param_offset) {
    VALUE_MAP[reduce_params[param_offset]->unique()] =
        mpdfg_params[param_offset + param_pointer];
    mpdfg
        ->data_residency[mpdfg_params[param_offset + param_pointer]->unique()] =
        Residency::Shared;
    param_offset += 1;
  }

  // std::cout << "VALUE Map Before parse:" << std::endl;
  // for(auto i : VALUE_MAP) {
  //   std::cout << i.first << ", " << i.second->debugName() << std::endl;
  // }

  if (update_graph.has_value()) {
    UPDATE_UDF = true;
    update_block = update_graph.value()->block();
    auto update_params = update_block->param_node()->outputs();
    size_t num_update_params = update_params.size();

    param_offset = 1;
    param_pointer = num_msg_params + num_reduce_params + 2;
    while (num_update_params > param_offset) {
      VALUE_MAP[update_params[param_offset]->unique()] =
          mpdfg_params[param_offset + param_pointer];
      mpdfg->data_residency[mpdfg_params[param_offset + param_pointer]
                                ->unique()] = Residency::Shared;
      param_pointer += 1;
    }
  }

  // message creation stage
  parse_stage(mpdfg, mpdfg_params, msg_block, Stage::Creation);
  std::cout << "After Merge Message MP-DFG=:" << std::endl;
  mpdfg->DFG->print(std::cout);

  VALUE_MAP.clear();
  VALUE_TO_BROADCAST.clear();
  // message aggregation stage
  parse_stage(mpdfg, mpdfg_params, reduce_block, Stage::Aggregation);
  if (UPDATE_UDF)
    parse_stage(mpdfg, mpdfg_params, update_block, Stage::Update);

  // post building optimization, redundant code elimination
  
  std::cout << "After Build MP-DFG=:" << std::endl;
  mpdfg->DFG->print(std::cout);

  dedup(mpdfg->DFG);
}
} // namespace graphiler
