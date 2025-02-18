# @time          : 2024/6/23 11:14
# @file          : eaug_base.py
# @Author        : wuqw
# @Description   :

import pickle
from servive import training_and_testing_eaug as train_eaug
from preprocess import eaug as pre_eaug
import json




if __name__ == "__main__":
    graphs = pre_eaug.get_graphs_with_coding_without_port()
    graph_dic = {"RS232": [], "s35932": [], "s38417": [], "s15850": [], "s38584": [], "EthernetMAC10GE": [], "B19": [],
                 "wb_conmax": []}
    for graph_cur in graphs:
        graph_base = graph_cur.fname.split("-")[0]
        graph_dic[graph_base].append(graph_cur)
    model_dic = dict()
    model_dic['model_save_root_path'] = "../data/cache/trained_model/eaug_base"
    model_dic['sample_root_path'] = "../data/sample"
    res_dic = dict()
    for test_key, test_value in graph_dic.items():
        model_dic['train_data_list'] = []
        model_dic['val_data_list'] = []
        for train_key, tranin_value in graph_dic.items():
            if train_key != test_key:
                model_dic['train_data_list'].extend(tranin_value)
            else:
                model_dic['val_data_list'].extend(test_value)
        model_dic['model_save_name'] = test_key
        res_model = train_eaug.train_grid(model_dic)
        res_dic[test_key] = res_model
        with open(model_dic['model_save_root_path'] + "/" + "res_dic.json", 'w') as ft:
            json.dump(res_dic, ft)

