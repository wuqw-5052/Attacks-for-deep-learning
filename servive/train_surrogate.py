# 作者：wuqw
# 时间：2023/11/3 10:56

from pojo import model as md
import pickle
import numpy as np
import dgl
import torch
import torch.nn.functional as F
import networkx as nx
from tool.earlystopping import EarlyStopping
from tool.evaluation import evaluation as Evaluation
import json
import os
import time
import random




# 参数声明
early_stop_num = 50
early_stop = False
test_flag = False
# is_break = False
path1 = "../data/cache/graph_with_feature/"
model_save_root_path = "../data/cache/trained_model/surrogate/"
sample_root_path = "../data/sample/surrogate/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_open_path = path1 + "train_without_port.pickle"
text_open_path = path1 + "test_without_port.pickle"

if not os.path.exists(os.path.dirname(model_save_root_path)):
    os.makedirs(os.path.dirname(model_save_root_path))



with open(train_open_path, "rb") as f:
    train_data_list = pickle.load(f)

with open(text_open_path, "rb") as f:
    val_data_list = pickle.load(f)

for val_data in val_data_list:
    for node in val_data['graph'].nodes(data=True):
        node_id, attrs = node
        if "feature" in attrs:
            new_feature = attrs["feature"][2:-2]
            val_data['graph'].nodes(data=True)[node_id]["feature"] = new_feature



feature_dim = len(list(train_data_list[0]['graph'].nodes._nodes.values())[0]['feature'])


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)



# 模型训练
def train(model, train_data_list, sample_path):
    model.train()
    tlosses = []
    for train_data in train_data_list:
        train_sample_path = sample_path + train_data.fname + ".pickle"
        if os.path.exists(train_sample_path):
            with open(train_sample_path, "rb") as f:
                train_subgraph_list = pickle.load(f)
        for sample in train_subgraph_list:
            subgraph = sample['subgraph']
            subgraph = subgraph.to_undirected()
            subgraph = nx.to_directed(subgraph)
            subgraph = dgl.add_self_loop(subgraph)
            subgraph = subgraph.to(device)
            batch_labels = subgraph.ndata['label']
            box_pred = model(subgraph)
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
    res = {}
    with torch.no_grad():
        for i, data in enumerate(val_data_list, 0):

            gv = data.graph
            gv = gv.to_undirected()
            gv = nx.to_directed(gv)
            gv = dgl.from_networkx(gv, node_attrs=['feature', 'label'])
            gv = dgl.add_self_loop(gv).to(device)
            batch_labels = gv.ndata['label']
            box_res = model(gv)
            box_res = F.log_softmax(box_res, dim=1)
            _, box_pred = box_res.max(dim=1)
            lst = Evaluation(box_pred, batch_labels)
            print(data['graphName'] + ";" +str(lst))
            res[data['graphName']] = lst
            #
    return res



if __name__ == "__main__":
    setup_seed(1234)
    hideNum_grid = [16, 32, 64]
    sample_split_num_grid = [5, 10, 15, 20]
    layerNum_grid = [2, 3]
    model_name = "GCN"
    epochs = 1000
    learning_rate = 0.01




    for sample_split_num in sample_split_num_grid:
        for layer_num in layerNum_grid:
            for hide_dim in hideNum_grid:
                try:
                    cur_grid = str(hide_dim) + "_" + str(layer_num) + "_" + str(sample_split_num)

                    model_savepath = model_save_root_path + cur_grid + ".pth"
                    sample_path = sample_root_path + "/" + "split_num_" + str(sample_split_num) + "/"
                    model = md.GCNModel(feature_dim, hide_dim, 2, layer_num).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    if early_stop_num > 0:  # 早停
                        early_stopping = EarlyStopping(patience=early_stop_num, verbose=True)
                    # 如果test_flag=True,则加载已保存的模型
                    if test_flag:
                        # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤
                        checkpoint = torch.load(model_savepath)
                        model.load_state_dict(checkpoint['model'])
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        epochs = checkpoint['epoch']
                        test(model, val_data_list)
                        continue

                        # 如果有保存的模型，则加载模型，并在其基础上继续训练
                    if os.path.exists(model_savepath):
                        checkpoint = torch.load(model_savepath)
                        model.load_state_dict(checkpoint['model'])
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        start_epoch = checkpoint['epoch']
                        self_state = checkpoint['early_stopping_state']
                        early_stop = self_state['early_stop']
                        early_stopping.load_self_state(self_state)
                        print('加载 epoch {} 成功！'.format(start_epoch))
                    else:
                        start_epoch = 0
                        print('无保存模型，将从头开始训练！')

                    for epoch in range(start_epoch+1, epochs):
                        if not early_stop:
                            optimizer.zero_grad()
                            tlosses = train(model, train_data_list, sample_path)
                            optimizer.step()  # 根据梯度更新网络参数
                            train_loss = np.mean(tlosses)
                            print("epoch: {} - train_loss: {}".format(epoch, train_loss))
                            if early_stop_num > 0:
                                early_stopping(train_loss, model, optimizer, epoch, model_savepath)
                                if early_stopping.early_stop:
                                    print("Epoch: {}. Early stopping.".format(epoch))
                                    early_stop = True

                            # 保存模型
                            # state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            #          'epoch': epoch}
                            # torch.save(state, model_savepath)
                        else:
                            res = test(model, val_data_list)
                            early_stop = False
                            # print(model_savepath + ";" + res)
                            with open(model_save_root_path + str(hide_dim) + "_" + str(
                                    layer_num) + ".json",
                                      'w') as ft:
                                json.dump(res, ft)
                            break

                except Exception as e:
                    print(str(e))
