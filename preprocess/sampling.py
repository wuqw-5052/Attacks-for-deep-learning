# @time          : 2024/4/29 11:48
# @file          : sampling.py
# @Author        : wuqw
# @Description   :

from preprocess import eaug as pre_eaug
import pickle
import networkx as nx
import dgl
import numpy.random as random
import os

is_EAUG = True
is_surrogate = False


def random_split_nomalgraph(n, normal_node_numbers, ht_index):
    numbers = list(range(1, normal_node_numbers))
    # 移除ht的id
    for num in ht_index:
        if num in numbers:
            numbers.remove(num)
    # 随机打乱列表中的数字顺序
    random.shuffle(numbers)
    # 计算每份应包含的数字数量，使用整除和取余来确定最后一份可能包含较少数字
    quotient, remainder = divmod(len(numbers), n)
    sizes = [quotient + 1 if i < remainder else quotient for i in range(n)]
    # 根据计算出的尺寸，将数字划分成n份
    start = 0
    split_numbers = []
    for size in sizes:
        end = start + size
        split_numbers.append(numbers[start:end])
        start = end
    return split_numbers


def random_sample(split_num, path_dic):

    distributed_root_path = path_dic['distributed_root_path']
    sample_root_path = path_dic['sample_root_path']
    open_path = path_dic['open_path']

    distributed_root_path = distributed_root_path + "split_num_"+str(split_num)+"/"
    sample_root_path = sample_root_path + "split_num_" + str(split_num) + "/"
    if not os.path.exists(os.path.dirname(distributed_root_path)):
        os.makedirs(os.path.dirname(distributed_root_path))

    if not os.path.exists(os.path.dirname(sample_root_path)):
        os.makedirs(os.path.dirname(sample_root_path))

    with open(open_path, "rb") as f:
        graphs = pickle.load(f)
    for j in range(len(graphs)):
        graph_object = {}
        graph = graphs[j].graph
        # 删除特殊节点
        port_gate_type = {"inPort", "outPort", "1'b0", "1'b1", "middle"}
        nodes_to_remove = [node[0] for node in graph.nodes(data=True) if node[1]['gateType'] in port_gate_type]
        graph.remove_nodes_from(nodes_to_remove)
        graph_name = graphs[j].fname
        node_name_to_int = {name: idx for idx, name in enumerate(graph.nodes())}
        int_to_node_name = {idx: name for name, idx in node_name_to_int.items()}
        graph_object["node_name_to_int"] = node_name_to_int
        graph_object["int_to_node_name"] = int_to_node_name
        dgl_graph = nx.relabel.relabel_nodes(graph, node_name_to_int)
        dgl_graph = dgl.from_networkx(dgl_graph, node_attrs=['feature', 'label'])
        if is_EAUG:
            # 添加反向边并设置边属性
            dgl_graph = pre_eaug.edge_adding_and_attr_setting(dgl_graph)
        distributed_path = distributed_root_path + graph_name + "/"
        sample_path = sample_root_path + graph_name+".pickle"
        ht_index = (dgl_graph.ndata['label'] == 1).nonzero(as_tuple=True)[0].tolist()
        normal_index = (dgl_graph.ndata['label'] == 0).nonzero(as_tuple=True)[0].tolist()
        normal_graph = dgl_graph.subgraph(normal_index)
        normal_graph_map_ids = normal_graph.ndata["_ID"].tolist()
        node_map, edge_map = dgl.distributed.partition_graph(
            g=normal_graph, graph_name=graph_name, num_parts=split_num,
            out_path=distributed_path, balance_edges=True, return_mapping=True)
        file_dir_name = os.path.join(distributed_path, graph_name+".json")
        subgraph_list = []
        isolated_list = []
        for part_i in range(split_num):
            subgraph_dic= {}
            g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = dgl.distributed.load_partition(
                file_dir_name, part_i)
            # inner_index_list = [index for index in range(len(g.ndata['inner_node'].tolist())) if g.ndata['inner_node'].tolist()[index] == 1]
            # inner_ID_list = g.ndata["_ID"].tolist()[0:len(inner_index_list)]
            inner_ID_list = g.ndata["_ID"].tolist()
            nomal_ori_ids = [node_map.tolist()[g_id] for g_id in inner_ID_list]
            ori_ids = [normal_graph_map_ids[nomal_ori_id] for nomal_ori_id in nomal_ori_ids]
            ori_ids.extend(ht_index)
            subgraph_with_ht = dgl_graph.subgraph(ori_ids)
            subgraph_dic["node_index"] = ori_ids
            subgraph_dic["subgraph"] = subgraph_with_ht
            subgraph_list.append(subgraph_dic)
            # 孤点处理 检查g图中是否存在inner_node的度为0的情况，如果有的话要单独处理，如果没有的话不用处理
            isolated_index_list = (g.in_degrees() == 0).nonzero(as_tuple=True)[0].tolist()
            # isolated_index_list = (subgraph_with_ht.in_degrees() == 0).nonzero(as_tuple=True)[0].tolist()
            if len(isolated_index_list) != 0:
                for isolated_index in isolated_index_list:
                    is_inner_node = g.ndata["inner_node"].tolist()[isolated_index]
                    if is_inner_node == 1:
                        g_id = g.ndata["_ID"].tolist()[isolated_index]
                        nomal_ori = node_map.tolist()[g_id]
                        ori = normal_graph_map_ids[nomal_ori]
                        print(graph_name + "子图" + str(part_i) + "---存在孤点---" + str(
                            len(isolated_index_list)) + "个-ori=" + str(ori) + "--label为" +
                              str(dgl_graph.ndata['label'].tolist()[ori]))

                        isolated_list.append(ori)
        # 处理孤点
        subgraph_list = merge_orphan_nodes(dgl_graph, subgraph_list, isolated_list, int_to_node_name)
        # normal_node_numbers = len(dgl_graph.nodes())
        # subgraph_normal_index_list = random_split_nomalgraph(split_num, normal_node_numbers, ht_index)
        # subgraph_dic_list = []
        # isolated_list = []
        # for subgraph_normal_index in subgraph_normal_index_list:
        #  子图存储
        with open(sample_path, "wb") as f1:
            pickle.dump(subgraph_list, f1)
        print(graph_name+"样本划分完成")


# 合并孤点到子图的函数
def merge_orphan_nodes(dgl_graph, subgraph_dic_list_1, isolated_list_1, int_to_node_name):
    for isolated_node in isolated_list_1:
        subgraph_to_merge = True
        for subgraph_dic in subgraph_dic_list_1:
            if subgraph_to_merge:
                # 检查孤点与子图之间是否有边
                # 使用 edge_ids 检查从孤点到子图节点的边
                ori_ids = subgraph_dic["node_index"]
                for ori_id in ori_ids:
                    hase_edge = dgl_graph.has_edges_between(isolated_node, ori_id)
                    if hase_edge:
                        subgraph_to_merge = False
                        # 合并孤点到子图
                        subgraph_dic["node_index"].append(isolated_node)
                        new_index = subgraph_dic["node_index"]
                        new_subgraph = dgl_graph.subgraph(new_index)
                        subgraph_dic["subgraph"] = new_subgraph
                        break
            else:
                break

        # 如果没有找到子图来合并孤点，则打印错误（理论上不应该发生，因为原图中没有孤点）
        if subgraph_to_merge:
            print(f"孤点 {int_to_node_name[isolated_node]} 没有找到可合并的子图，但原图不应包含孤点")
    return subgraph_dic_list_1


if __name__ == "__main__":

    path_dic = dict()

    if is_EAUG:
        path_dic['distributed_root_path'] = "../data/sample/distributed_partition_graph/"
        path_dic['sample_root_path'] = "../data/sample/"
        path_dic['open_path'] = "../data/cache/graphs_with_code/eaug_graphs.pickle"
    elif is_surrogate:
        path_dic['distributed_root_path'] = "../data/sample/surrogate/distributed_partition_graph/"
        path_dic['sample_root_path'] = "../data/sample/surrogate/"
        path_dic['open_path'] = "../data/cache/graphs_with_code/surrogate_graphs.pickle"


    split_num_list = [5, 10, 15, 20]
    for split_num in split_num_list:
        random_sample(split_num, path_dic)
