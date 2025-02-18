# 作者：wuqw
# 时间：2023/10/26 21:25
import random
import os
import pickle
import torch
from preprocess import netlist_message as message
from preprocess import one_hot as coding
import networkx as nx
import dgl
import numpy as np
from pathlib import Path


# def count_disconnected_subgraphs():
#     for fname in files:
#         graph_path = "../data/cache/ngraph/" + fname + ".pickle"
#         with open(graph_path, "rb") as f:
#             graph_object = pickle.load(f)
#         graph = graph_object.graph
#         visited = set()
#         subgraph_count = 0
#         for node in graph:
#             if node not in visited:
#                 # 使用递归DFS来标记当前子图中的所有节点为已访问
#                 nx.dfs_preorder_nodes(graph, source=node)
#                 # 因为dfs_preorder_nodes不直接支持visited集合，我们需要手动添加
#                 for n in nx.dfs_preorder_nodes(graph, source=node):
#                     visited.add(n)
#                     # 发现了一个新的子图
#                 subgraph_count += 1
#         # print(f"The number of disconnected subgraphs is: {num_subgraphs}")

def get_netlist_graph_obj(graph_path):
    with open(graph_path, "rb") as f:
        module_graph_obj = pickle.load(f)
    return module_graph_obj

def eaug_coding(netlist_graph_obj):
    min_distances_to_inport, min_distances_to_outport = message.get_min_distances_to_inport_and_outport_dic(netlist_graph_obj)
    port_gate_type = {"inPort", "outPort", "1'b0", "1'b1", "middle"}
    one_hot_encoded = coding.base_onehot()  # 获得的one-hot是基于基编码

    for node in netlist_graph_obj.graph.nodes(data=True):
        feature = list()
        if node[1]['gateType'] in port_gate_type:
            # 特殊节点不编码
            continue
        gate_type_base = message.get_gate_type_base(node)
        degree = message.get_node_indegree_and_outdegree(gate_type_base)
        feature.extend(degree)
        dis = message.get_min_distances_to_inport_and_outport(min_distances_to_inport, min_distances_to_outport, node)
        feature.extend(dis)
        feature.extend(list(one_hot_encoded[gate_type_base]))

        probability = message.get_node_probability(gate_type_base)
        feature.extend(probability)

        node[1]["feature"] = feature

    # label = torch.tensor([i['label'] for i in module_graph_obj.graph.nodes._nodes.values()])
    # j = 0
    # for i in label:
    #     if i > 0:
    #         j += 1
    # weigh_ratio = torch.tensor([round(j / len(label), 4), 1.0])
    return netlist_graph_obj

def all_files_with_eaug_coding():

    """
    用基编码，没有去除端口节点。
    """
    path1 = "../data/cache/graphs_with_code/"
    if not os.path.exists(os.path.dirname(path1)):
        os.makedirs(os.path.dirname(path1))
    netlist_path = Path("../data/netlist_graph/eaug/")
    graphs = list()
    for item in netlist_path.iterdir():
        netlist_graph_obj = get_netlist_graph_obj(item)
        netlist_graph_obj.fname = item.stem
        netlist_graph_obj = eaug_coding(netlist_graph_obj)
        graphs.append(netlist_graph_obj)
        with open(path1 + "eaug_graphs.pickle", 'wb') as f:
            pickle.dump(graphs, f)


def edge_adding_and_attr_setting(dglGraph):
    """
    :param dglGraph:
    :return:
    """
    edge_feature = torch.tensor([[0, 1]], dtype=torch.float32)
    # 扩展这个张量以匹配边的数量
    edge_features = edge_feature.repeat(dglGraph.number_of_edges(), 1)
    dglGraph.edata['feat'] = edge_features
    # 添加反向边并设置特征为[1, 0]
    # 反转边的方向
    src, dst = dglGraph.edges()
    reversed_src, reversed_dst = dst, src
    # 创建一个布尔掩码，用于标记哪些反向边是原本就存在的
    new_edges_mask = ~dglGraph.has_edges_between(reversed_src, reversed_dst)
    # 获取新的反向边的索引
    new_reversed_src, new_reversed_dst = reversed_src[new_edges_mask], reversed_dst[new_edges_mask]
    # 将反向边添加到图中
    dglGraph.add_edges(new_reversed_src, new_reversed_dst)
    # 初始化反向边的特征，每条边都是[1, 0]这个二维向量
    reversed_edge_feature = torch.tensor([[1, 0]], dtype=torch.float32)
    # 获取新添加的边的索引（即反向边的索引）
    # 注意：这里使用graph.number_of_edges()可能不准确，因为新添加的边数量是new_reversed_src的长度
    reversed_edge_ids = torch.arange(len(src), len(src) + len(new_reversed_src))
    # 设置新添加的边的特征
    dglGraph.edata['feat'][reversed_edge_ids] = reversed_edge_feature.repeat(len(reversed_edge_ids), 1)
    return dglGraph


def networkx_to_dgl(graph):
    node_name_to_int = {name: idx for idx, name in enumerate(graph.nodes())}
    # int_to_node_name = {idx: name for name, idx in node_name_to_int.items()}
    dgl_graph = nx.relabel.relabel_nodes(graph, node_name_to_int)
    dgl_graph = dgl.from_networkx(dgl_graph, node_attrs=['feature', 'label'])
    # 添加反向边并设置边属性
    dgl_graph = edge_adding_and_attr_setting(dgl_graph)
    return dgl_graph

def setup_seed(seed):
    #
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

def get_graph_with_coding_without_port(netlist_graph_obj):
    port_gate_type = {"inPort", "outPort", "1'b0", "1'b1", "middle"}
    nodes_to_remove = [node[0] for node in netlist_graph_obj.graph.nodes(data=True) if node[1]['gateType'] in port_gate_type]
    netlist_graph_obj.graph.remove_nodes_from(nodes_to_remove)
    return netlist_graph_obj


def get_graphs_with_coding_without_port():
    netlist_graph_objs = get_graphs_with_coding_with_port()
    for netlist_graph_obj in netlist_graph_objs:
        graph = netlist_graph_obj.graph
        port_gate_type = {"inPort", "outPort", "1'b0", "1'b1", "middle"}
        nodes_to_remove = [node[0] for node in graph.nodes(data=True) if node[1]['gateType'] in port_gate_type]
        graph.remove_nodes_from(nodes_to_remove)
    return netlist_graph_objs

def get_graphs_with_coding_with_port():
    open_path = "../data/cache/graphs_with_code/eaug_graphs.pickle"
    with open(open_path, "rb") as f:
        netlist_graph_objs = pickle.load(f)
    return netlist_graph_objs



if __name__ == "__main__":
    all_files_with_eaug_coding()


