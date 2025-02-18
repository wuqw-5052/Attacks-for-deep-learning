# -*- coding: UTF-8 -*-
'''
@File    ：NetlistGraph.py
@Author  ：wuqw
@Date    ：2024/11/22 10:44:40
'''

class NetlistGraph:
    def __init__(self, fname, graph):
        self.graph = graph
        self.fname = fname

    def set_graph(self, graph):
        self.graph =graph

    def get_graph(self):
        return self.graph

    def set_fname(self, fname):
        self.fname = fname

    def get_fname(self):
        return self.fname

