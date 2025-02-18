# 作者：wuqw
# 时间：2024/6/21 16:01

from pojo import model as md
from e_attack import logically_equivalent as lq
import pickle
import dgl
import torch
import torch.nn.functional as F
import networkx as nx
import math
import random
from tool.evaluation import evaluation as Evaluation

# 参数设置
hide_Num = 16
layer_Num = 3
replace_time_list = [0.2,0.4,0.6,0.8,1]



surrogate_files = ['s15850-T100','s38584-T100', 's38584-T200', 's38584-T300',
         'EthernetMAC10GE-T700', 'EthernetMAC10GE-T710', 'EthernetMAC10GE-T720',
         'EthernetMAC10GE-T730', 'B19-T100', 'B19-T200', 'wb_conmax-T100', 'VGA-LCD-T100', 'AES-T2200']

files_1 = ['RS232-T1100', 'RS232-T1000', 'RS232-T1200', 'RS232-T1300', 'RS232-T1400', 'RS232-T1500',
           'RS232-T1600', 's35932-T100', 's35932-T200', 's35932-T300', 's38417-T100',
           's38417-T200', 's38417-T300']

path1 = "../data/cache/surrogate/graph_with_feature/"
text_open_path = path1 + "test_without_port.pickle"

surrogate_model_root_path = "../data/cache/surrogate/trained_model/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open(text_open_path, "rb") as f:
    graphs = pickle.load(f)

open_path_with_port = path1 + "with_port.pickle"
with open(open_path_with_port, "rb") as fs:
    graphs_with_port = pickle.load(fs)

# train_data_list = []
# val_data_list = []
# for index in range(len(graphs)):
#     if graphs[index]["graphName"] not in files_1:
#         train_data_list.append(graphs[index])
#     else:
#         val_data_list.append(graphs[index])



graphs_with_port_list = {}
for index in range(len(graphs_with_port)):
    graphName = graphs_with_port[index]["graphName"]
    if graphName  in files_1:
        graphs_with_port_list[graphName]=graphs_with_port[index]['graph']



def load_surrogate_model():
    """
    加载代理模型
    """
    feature_dim = len(list(graphs[0]['graph'].nodes._nodes.values())[0]['feature'])
    model_savepath = surrogate_model_root_path + str(hide_Num) + "_" + str(layer_Num) + ".pth"
    checkpoint = torch.load(model_savepath)
    model = md.GCNModel(feature_dim, hide_Num, 2, layer_Num).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model




def getTP(g, model):


    gv = nx.DiGraph(g)
    gv = gv.to_undirected()
    gv = nx.to_directed(gv)
    gv = dgl.from_networkx(gv, node_attrs=['feature', 'label'])
    gv = dgl.add_self_loop(gv).to(device)
    batch_labels = gv.ndata['label']
    with torch.no_grad():  # 禁用梯度计算，减少内存消耗和加速计算
        box_res = model(gv)
    box_res = F.log_softmax(box_res, dim=1)
    _, box_pred = box_res.max(dim=1)
    lst = Evaluation(box_pred, batch_labels)
    min_tp = lst[4]
    return min_tp



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






def dynamic_adversary_sample(val_data_list):
    model = load_surrogate_model()
    for i in range(len(val_data_list)):
        # if val_data_list[i]["graphName"] not in ['RS232-T1100', 's35932-T200']:
        #     continue
        for replace_time in replace_time_list:
            # 当前网表
            g_curr = val_data_list[i]
            # 原始图
            gv = g_curr['graph']

            # 带端口的图
            graph_with_port = graphs_with_port_list[g_curr["graphName"]]


            # 获取N（Vt）
            N_Vt = get_NVt(gv)
            iter_num = int(len(N_Vt) * replace_time)
            i = int(pow(math.comb(len(N_Vt),iter_num), 1/3))


            g_now = nx.DiGraph(gv)

            tp_value_min = float('inf')
            result_save_path = "../data/result/" + g_curr["graphName"] + "_" + str(replace_time) # replace_time
            stop = 500
            j = 0

            # 随机取个结点，随机取几次？
            while i != 0:

                j = j + 1
                if j > stop:
                    print("j大于stop；j="+str(j))
                    break
                iter_Vt = random.sample(N_Vt, iter_num)

                g_temp = nx.DiGraph(g_now)

                g_temp_with_port = nx.DiGraph(graph_with_port)


                for item in iter_Vt:
                    g_temp = lq.lq_classfiy(g_temp, item)
                    g_temp_with_port = lq.lq_classfiy(g_temp_with_port, item)

                tp_value = getTP(g_temp, model)
                i = i - 1
                print("i==" + str(i))
                if tp_value_min == 0:
                    print("tp_value_min=0;break")
                    break

                if tp_value < tp_value_min:
                    print(result_save_path + ";" + str(tp_value) + ";" + "j:" + str(j))
                    j = 0
                    tp_value_min = tp_value

                    with open(result_save_path + ".pickle", 'wb') as f:
                        pickle.dump(g_temp_with_port, f)



def static_adversary_sample(val_data_list):
    print()

if __name__ == "__main__":
    static_adversary_sample












