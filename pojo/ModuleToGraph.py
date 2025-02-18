# 作者：wuqw
# 时间：2023/10/27 15:37
import networkx as nx;


class ModuleToGraph:

    def __init__(self, moudleName):
        self.moudleName = moudleName
        self.inputNode = {}
        self.outputNode = {}
        self.graph = nx.DiGraph()
        self.moduleParamTuple = tuple
        self.fname = ""
        self.weigh_ratio = 0.0

