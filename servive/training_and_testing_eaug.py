# @time          : 2024/4/26 19:16
# @file          : training_and_testing_eaug.py
# @Author        : wuqw
# @Description   :
import pickle
import numpy as np
import dgl
import torch
import torch.nn.functional as F
import pojo.eaug_model as getModel
from tool.earlystopping import EarlyStopping
from tool.evaluation import evaluation as Evaluation
import json
import os
from preprocess import eaug as pre_eaug




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_grid(model_dic):
    """
    :param model_dic: [model_save_root_path, model_save_name, val_data_list]
    :return:
    """
    hideNum_grid = [16, 32]
    layerNum_grid = [2, 3]  #
    sample_split_num_grid = [5, 10, 15, 20]  #
    model_grid = ["GAT", "GIN", "MPNN"]
    res_model = dict()
    for model_str in model_grid:
        res_model[model_str] = dict()
        for layer_num in layerNum_grid:
            for hide_dim in hideNum_grid:
                    for sample_split_num in sample_split_num_grid:
                        cur_grid = str(hide_dim) + "_" + str(layer_num) + "_" + str(sample_split_num)
                        model_dic["layer_num"] = layer_num
                        model_dic["hide_dim"] = hide_dim
                        model_dic["sample_split_num"] = sample_split_num
                        model_dic["model_str"] = model_str
                        res = training_and_testing(model_dic, False)
                        res_model[model_str][cur_grid] = res
    return res_model



def training_and_testing(model_dic, test_flag):
    early_stop_num = 50
    learning_rate = 0.01
    epochs = 1000
    pre_eaug.setup_seed(1234)
    layer_num = model_dic["layer_num"]
    hide_dim = model_dic["hide_dim"]
    sample_split_num = model_dic["sample_split_num"]
    model_str = model_dic["model_str"]
    val_data_list = model_dic['val_data_list']
    model_save_root_path = model_dic['model_save_root_path']
    model_save_name = model_dic['model_save_name']
    feature_dim = len(list(val_data_list[0].graph.nodes(data=True))[0][1]['feature'])
    if model_str == "MPNN":
        model = getModel.MPNN(feature_dim, hide_dim, 2, layer_num, edge_dim=2).to(device)
    elif model_str == "GIN":
        model = getModel.GIN(feature_dim, hide_dim, 2, layer_num, edge_dim=2).to(device)
    elif model_str == "GAT":
        model = getModel.GAT(feature_dim, hide_dim, 2, layer_num, edge_dim=2).to(device)

    cur_grid = str(hide_dim) + "_" + str(layer_num) + "_" + str(sample_split_num)
    model_save_path = model_save_root_path + "/" + model_str + "/"
    model_save_path = model_save_path + cur_grid + "/"
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))
    model_save_path = model_save_path + model_save_name + ".pth"
    if test_flag:
        # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤
        checkpoint = torch.load(model_save_path)
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # epochs = checkpoint['epoch']
        res = test(model, val_data_list)
        return res
    try:
        train_data_list = model_dic['train_data_list']
        sample_root_path = model_dic['sample_root_path']
        sample_path = sample_root_path + "/" + "split_num_" + str(sample_split_num) + "/"
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        if early_stop_num > 0:  # 启动早停
            early_stop_obj = EarlyStopping(patience=early_stop_num, verbose=True)
            # 如果有保存的模型，则加载模型，并在其基础上继续训练
        if os.path.exists(model_save_path):
            checkpoint = torch.load(model_save_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            self_state = checkpoint['early_stopping_state']
            early_stop_obj.load_self_state(self_state)
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            start_epoch = 0
            print('无保存模型，将从头开始训练！')
        for epoch in range(start_epoch+1, epochs):
            if not early_stop_obj.early_stop:
                optimizer.zero_grad()
                tlosses = train(train_data_list, sample_path, model)
                optimizer.step()  # 根据梯度更新网络参数
                train_loss = np.mean(tlosses)

                print("epoch: {} - train_loss: {}".format(epoch, train_loss))
                if early_stop_num > 0:
                    early_stop_obj(train_loss, model, optimizer, epoch, model_save_path)
                    if early_stop_obj.early_stop:
                        print("Epoch: {}. Early stopping.".format(epoch))
            else:
                print(model_save_path)
                break
        res = test(model, val_data_list)
    except Exception as e:
        print(str(e))

    return res

def train(train_data_list, sample_path, model):
    model.train()
    tlosses = []
    for train_data in train_data_list:
        train_sample_path = sample_path + train_data.fname + ".pickle"

        if os.path.exists(train_sample_path):
            with open(train_sample_path, "rb") as f:
                train_subgraph_list = pickle.load(f)

        for sample in train_subgraph_list:
            subgraph = sample['subgraph']
            subgraph = dgl.add_self_loop(subgraph)
            subgraph = subgraph.to(device)
            batch_labels = subgraph.ndata['label']
            box_pred = model(subgraph, device)
            # box_pred = F.softmax(box_pred, dim=1)
            box_pred = F.log_softmax(box_pred, dim=1)
            lossHT = F.cross_entropy(box_pred, batch_labels,
                                     # weight=train_data[i]['weigh_ratio'].cuda(),
                                     reduction='none')
            loss = lossHT
            loss = loss.mean()
            loss.backward()  # 计算当前样本的梯度，不断累加loss.backward()#计算当前样本的梯度，不断累加
            tlosses.append(loss.item())
    return tlosses


def test(model, val_data_list):
    model.eval()
    res = []
    with torch.no_grad():
        for i, data in enumerate(val_data_list, 0):
            val_graph = data.graph
            val_graph = pre_eaug.networkx_to_dgl(val_graph)
            val_graph = dgl.add_self_loop(val_graph)
            batch_labels = val_graph.ndata['label']
            box_res = model(val_graph, device)
            box_res = F.log_softmax(box_res, dim=1)
            _, box_pred = box_res.max(dim=1)
            lst = Evaluation(box_pred, batch_labels)
            res.append(lst)
    return res


if __name__ == "__main__":
    graphs = pre_eaug.get_graphs_with_coding_without_port()

    model_dic = dict()
    model_dic['model_save_root_path'] = "../data/cache/trained_model/eaug"
    model_dic['sample_root_path'] = "../data/sample"
    res_dic = dict()
    for val_index in range(len(graphs)):
        model_dic['train_data_list'] = []
        model_dic['val_data_list'] = []
        for train_index in range(len(graphs)):
            if train_index != val_index:
                model_dic['train_data_list'].append(graphs[train_index])
            else:
                model_dic['val_data_list'].append(graphs[val_index])
        model_dic['model_save_name'] = graphs[val_index].fname
        res_model = train_grid(model_dic)
        res_dic[model_dic['model_save_name']] = res_model
        with open(model_dic['model_save_root_path'] + "/" + "res_dic.json", 'w') as ft:
            json.dump(res_dic, ft)



