# -*- coding: UTF-8 -*-
'''
@File    ：surrogate.py
@Author  ：wuqw
@Date    ：2024/11/29 16:50:07
'''
from pathlib import Path
import os
from preprocess import eaug as pre_eaug
from preprocess import one_hot as coding
from preprocess import netlist_message as message
import torch
import pickle



def training_data_coding(netlist_path, save_path):
    graphs = list()
    for item in netlist_path.iterdir():
        netlist_graph_obj = pre_eaug.get_netlist_graph_obj(item)
        netlist_graph_obj.fname = item.stem
        netlist_graph_obj = coding(netlist_graph_obj)
        graphs.append(netlist_graph_obj)
        with open(save_path, 'wb') as f:
            pickle.dump(graphs, f)




def coding(netlist_graph_obj):

    port_gate_type = {"inPort", "outPort", "1'b0", "1'b1", "middle", "1'h1"}
    one_hot_encoded = coding.base_onehot()  # 获得的one-hot是基于基编码
    for node in netlist_graph_obj.graph.nodes(data=True):
        feature = list()
        try:
            if node[1]['gateType'] in port_gate_type:
                # 特殊节点不编码
                continue
        except KeyError:
            print(str(node[0]))
        gate_type_base = message.get_gate_type_base(node)
        if gate_type_base == "DFFNAR":
            node[1]['gateType'] == "DFFARX1"
            gate_type_base = "DFFAR"
            print(netlist_graph_obj.fname +"-----"+str(node[0]))
        feature.extend(list(one_hot_encoded[gate_type_base]))
        node[1]["feature"] = feature
    label = torch.tensor([i['label'] for i in netlist_graph_obj.graph.nodes._nodes.values()])
    j = sum(1 for i in label if i > 0)
    weigh_ratio = torch.tensor([round(j / len(label), 4), 1.0])
    netlist_graph_obj.weigh_ratio = weigh_ratio
    return netlist_graph_obj


if __name__ == "__main__":
    save_path = "../data/cache/graphs_with_code/" + "surrogate_testing_graphs.pickle"
    netlist_path = Path("../data/netlist_graph/eaug/")
    training_data_coding(netlist_path, save_path)