# -*- coding: UTF-8 -*-
'''
@File    ：one_hot.py
@Author  ：wuqw
@Date    ：2024/11/14 21:16:29
'''

import json
import numpy as np

def base_onehot():
    all_gate_type_path = "../data/dic/all_gate_type.json"
    with open(all_gate_type_path, 'r') as f:
        all_gate_type = json.load(f)
    base_type = list(all_gate_type.keys())
    # 为每个键创建one-hot编码
    base_onehot_encodings = {type: create_onehot_encoding(base_type, type) for type in base_type}
    return base_onehot_encodings


def create_onehot_encoding(keys, key_to_encode):
    # 确定one-hot编码的维度
    num_classes = len(keys)
    # 初始化一个全0的向量
    onehot_vector = np.zeros(num_classes, dtype=np.int32)
    # 将对应键的位置设为1
    onehot_vector[keys.index(key_to_encode)] = 1
    return onehot_vector

