# -*- coding: UTF-8 -*-
'''
@File    ：random_attack.py
@Author  ：wuqw
@Date    ：2024/11/21 18:02:09
'''
from preprocess import eaug as pre_eaug
import random
from e_attack import logically_equivalent as leq
from servive import training_and_testing_eaug as tt_eaug
from pojo import ModuleToGraph as mp
from tool import excel as ex


def get_NVt(gv):
    troNodes = [key for key, value in gv.nodes(data=True) if value['label'] == 1]
    N_Vt = set()
    N_Vt.update(troNodes)
    for i in range(len(troNodes)):
        pre_of_node = [node for node in gv if gv.has_edge(node, troNodes[i])]
        if len(pre_of_node) == 0:
            continue
        else:
            N_Vt.update(pre_of_node)
        post_of_node = [node for node in gv if gv.has_edge(troNodes[i], node)]
        if len(post_of_node) == 0:
            continue
        else:
            N_Vt.update(post_of_node)
    return N_Vt

def random_adversarial_sample_generation(graph, iter_num):
    if iter_num >= 0:
        N_Vt = get_NVt(graph)
        iter_Vt = random.sample(N_Vt, iter_num)
        g_temp = graph
        for item in iter_Vt:
            g_temp = leq.lq_classfiy(g_temp, item)
        return g_temp
    else:
        return graph



def get_results_of_AS_on_eaug(netlist_graph_obj, rate):
    len_NVt = len(get_NVt(netlist_graph_obj.graph))
    iter_num =int(len_NVt * rate)
    AS_obj = mp.ModuleToGraph("uart")
    AS_obj.graph = random_adversarial_sample_generation(netlist_graph_obj.graph, iter_num)
    AS_obj.fname = netlist_graph_obj.fname+"_as_"
    AS_obj = pre_eaug.eaug_coding(AS_obj)
    AS_obj = pre_eaug.get_graph_with_coding_without_port(AS_obj)
    return AS_obj




def base_testing_on_trained_model():
    base_dic = {"RS232": ['RS232-T1100', 'RS232-T1000', 'RS232-T1200', 'RS232-T1300', 'RS232-T1400', 'RS232-T1500','RS232-T1600'],
                "s35932": ['s35932-T100', 's35932-T200', 's35932-T300'],
                "s38417": ['s38417-T100','s38417-T200', 's38417-T300'],
                "s15850": ['s15850-T100'],
                "s38584": ['s38584-T100', 's38584-T200', 's38584-T300'],
                "EthernetMAC10GE": ['EthernetMAC10GE-T700', 'EthernetMAC10GE-T710', 'EthernetMAC10GE-T720'],
                "B19": ['B19-T100', 'B19-T200'],
                 "wb_conmax": ['wb_conmax-T100']}
    model_dic = dict()
    model_dic['model_save_root_path'] = "../data/cache/trained_model/eaug_base"
    model_dic['layer_num'] = 2
    model_dic['hide_dim'] = 32
    model_dic['sample_split_num'] = 15
    model_dic['model_str'] = "GAT"
    res_base_dic = dict()
    for key, value in base_dic.items():
        model_dic['model_save_name'] = key
        res_rate_dic = dict()
        iter_nums_rate = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        for rate in iter_nums_rate:
            model_dic['val_data_list'] = list()
            model_dic['rate'] = rate
            for file in value:
                file_path= "../data/netlist_graph/eaug/"+file+".pickle"
                netlist_graph_obj = pre_eaug.get_netlist_graph_obj(file_path)
                netlist_graph_obj.fname = file
                AS_obj = get_results_of_AS_on_eaug(netlist_graph_obj, rate)
                model_dic['val_data_list'].append(AS_obj)
            res_files = tt_eaug.training_and_testing(model_dic, True)
            res_rate = ex.calculate_column_averages(res_files)
            res_rate_dic[str(rate)] = res_rate
        res_base_dic[key] = res_rate_dic
        print(res_base_dic)

if __name__ == "__main__":
    base_testing_on_trained_model()

