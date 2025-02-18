# -*- coding: UTF-8 -*-
'''
@File    ：netlist_message.py
@Author  ：wuqw
@Date    ：2024/11/14 21:24:17
'''

import os
import json
import networkx as nx
from pyverilog.vparser.ast import *
from pyverilog.vparser.parser import parse
from pathlib import Path

files = ['RS232-T1100', 'RS232-T1000', 'RS232-T1200', 'RS232-T1300', 'RS232-T1400', 'RS232-T1500',
             'RS232-T1600', 's35932-T100', 's35932-T200', 's35932-T300', 's38417-T100',
             's38417-T200', 's38417-T300', 's15850-T100','s38584-T100', 's38584-T200', 's38584-T300',
             'EthernetMAC10GE-T700', 'EthernetMAC10GE-T710',
             'EthernetMAC10GE-T720', 'EthernetMAC10GE-T730', 'B19-T100', 'B19-T200', 'wb_conmax-T100']
rawpath = "../data/raw_data/"

def get_all_trojan_node():
    all_trojan_node = {}
    for fname in files:
        fpath = rawpath + fname + "-HT.txt"
        fpath_trojan_node_List = []
        # 读取 HT
        with open(fpath, 'r') as f:
            HTlines = f.readlines()  # 读取文件所有行
            for line in HTlines:
                plit_list = line.split(" ")
                trojan_node_name = plit_list[1]
                fpath_trojan_node_List.append(trojan_node_name)
        all_trojan_node[fname] = fpath_trojan_node_List
    path = "../data/dic/all_trojan_node.json"
    with open(path, 'w') as ft:
        json.dump(all_trojan_node, ft)
    return all_trojan_node


def get_all_net_type():
    netTypeSet = set()
    for fname in files:
        nl_fpath = rawpath + fname + ".v"
        ast, directives = parse([nl_fpath])
        for module in ast.description.definitions:
            if isinstance(module, ModuleDef):
                for instanceListItem in module.items:
                    if isinstance(instanceListItem, InstanceList):
                        for instance in instanceListItem.instances:
                            for portDuan in instance.portlist:
                                netType = portDuan.portname
                                netTypeSet.add(netType)
    # 轻易不替换allNetType.txt文件，里面的数据经过处理
    path = "../data/dic/all_net_type1.txt"
    file = open(path, 'w')
    file.write("netTypeSet"+str(netTypeSet))
    file.close()

def get_all_gate_type():
    allGateTypeAndNetType = {}
    surrogate_path_root = "../data/dataset/yosys/"
    path = Path(surrogate_path_root)
    for item in path.iterdir():
        for path_graph in item.iterdir():
            ast, directives = parse([path_graph])
            for module in ast.description.definitions:
                if isinstance(module, ModuleDef):
                    for instanceListItem in module.items:
                        if isinstance(instanceListItem, InstanceList):
                            gateType = instanceListItem.module  # 逻辑单元类型
                            if gateType == 'MX2X1':
                                print(path_graph)
                                return
                            index_X = gateType.find('X')  #XNOR
                            gate_type_base = gateType[0:index_X]
                            if gate_type_base in allGateTypeAndNetType:
                                allGateTypeAndNetType[gate_type_base].add(gateType)
                            else:
                                allGateTypeAndNetType[gate_type_base] = set()
                                allGateTypeAndNetType[gate_type_base].add(gateType)
    path_save = "../data/dic/surrogate_all_gate_type.json"
    for gate_type_base in allGateTypeAndNetType:
        allGateTypeAndNetType[gate_type_base] = list(allGateTypeAndNetType[gate_type_base])
    with open(path_save, 'w') as ft:
        json.dump(allGateTypeAndNetType, ft)

def get_node_indegree_and_outdegree(gate_type_base):

    degree = []
    gate_indegree_path = "../data/dic/gate_indegree.json"
    with open(gate_indegree_path, 'r') as f:
        gate_indegree_dic = json.load(f)

    degree.append(gate_indegree_dic[gate_type_base])

    if gate_type_base in ["SDFF", "DFFN", "DFF", "HADD", "DFFAR", "DFFAS",
                          "FADD", "SDFFAS", "SDFFAR", "SDFFASR", "SDFFSR", "RDFFN"]:
        out_degree = 2
    else:
        out_degree = 1
    degree.append(out_degree)
    return degree

def get_node_probability(gate_type_base):


    gate_probability_path = "../data/dic/gate_probability.json"
    with open(gate_probability_path, 'r') as f:
        gate_probability_dic = json.load(f)

    return gate_probability_dic[gate_type_base]

def get_min_distances_to_inport_and_outport_dic(netlist_graph_obj):
    graph = netlist_graph_obj.graph
    fname = netlist_graph_obj.fname

    distance_path = "../data/dic/distance_to_port/with_port_" + fname + ".json"

    is_as = fname.endswith("_as_")


    if not is_as and os.path.exists(distance_path):
        with open(distance_path, 'r') as f:
            min_distances_json = json.load(f)
        min_distances_to_outport = min_distances_json["min_distances_to_outport"]
        min_distances_to_inport = min_distances_json["min_distances_to_inport"]
    elif is_as and os.path.exists(distance_path):
        print()


    elif not is_as and os.path.exists(distance_path) :
        min_distances_to_outport = {node: float('inf') for node in graph
                                    if graph.nodes[node]['gateType'] != 'outPort'}
        min_distances_to_inport = {node: float('inf') for node in graph
                                   if graph.nodes[node]['gateType'] not in ["inPort", "1'b0", "1'b1"]}

        # 计算到输出端口和输入端口的最短距离,这里耗时长,存储起来
        for node in graph.nodes(data=True):
            gate_type = node[1]['gateType']
            if gate_type in ["inPort", "1'b0", "1'b1"]:
                inport_distances = nx.single_source_shortest_path_length(graph, node[0])

                for node_to, dist_to in inport_distances.items():
                    if graph.nodes[node_to]['gateType'] not in ["inPort", "1'b0", "1'b1"] \
                            and dist_to < min_distances_to_inport[node_to]:
                        min_distances_to_inport[node_to] = dist_to

                min_distances_to_inport[node[0]] = 0

            if gate_type != "outPort":
                outport_distances = nx.single_source_shortest_path_length(graph, node[0])
                for to_node, to_dist in outport_distances.items():
                    if graph.nodes[to_node]['gateType'] == 'outPort' and to_dist < min_distances_to_outport[node[0]]:
                        min_distances_to_outport[node[0]] = to_dist
            else:
                min_distances_to_outport[node[0]] = 0

        if not os.path.exists(os.path.dirname(distance_path)):
            os.makedirs(os.path.dirname(distance_path))

        with open(distance_path, 'w') as ft:
            json.dump({"min_distances_to_outport": min_distances_to_outport,
                       "min_distances_to_inport": min_distances_to_inport}, ft)

    return min_distances_to_inport, min_distances_to_outport

def get_min_distances_to_inport_and_outport(min_distances_to_inport, min_distances_to_outport, node_cur):
    dis = []
    if min_distances_to_inport[node_cur[0]] == float('inf'):
        dis.append(-1)  # 存在某个逻辑门的输入没有接入任何其他逻辑门的情况
    else:
        dis.append(min_distances_to_inport[node_cur[0]])

    if min_distances_to_outport[node_cur[0]] == float('inf'):
        dis.append(-1)  # 存在某个逻辑门的输出没有接入任何其他逻辑门的情况
    else:
        dis.append(min_distances_to_outport[node_cur[0]])
    return dis

def get_gate_type_base(node):
    gate_type = node[1]['gateType']
    first_index = gate_type.find('X')
    second_index = gate_type.find('X', first_index + 1)
    if second_index == -1:
        # 如果没有找到第二个X，返回第一个X的下标
        index = first_index
    else:
        # 如果找到了第二个X，返回它的下标
        index = second_index
    if gate_type == "MX2X1":
        gate_type_base = "MUX21"
    else:
        if index == -1:
            gate_type_base = gate_type
        else:
            gate_type_base = gate_type[0:index]
    return gate_type_base

if __name__ == "__main__":
    get_all_gate_type()