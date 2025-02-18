# 作者：wuqw
# 时间：2023/10/26 11:32
from pyverilog.vparser.ast import *
from pyverilog.vparser.parser import parse
import os
import networkx as nx
import pojo.ModuleToGraph as mp
import json
import pickle

# files = ['RS232-T1100', 'RS232-T1000', 'RS232-T1200', 'RS232-T1300', 'RS232-T1400', 'RS232-T1500',
#          'RS232-T1600', 's15850-T100', 's35932-T100', 's35932-T200', 's35932-T300', 's38417-T100',
#          's38417-T200', 's38417-T300', 's38584-T100', 's38584-T200', 's38584-T300',
#          'EthernetMAC10GE-T700', 'EthernetMAC10GE-T710', 'EthernetMAC10GE-T720',
#          'EthernetMAC10GE-T730', 'B19-T100', 'B19-T200', 'wb_conmax-T100', 'AES-T2200', 'VGA-LCD-T100']

# files90 = ['RS232-T1100', 'RS232-T1000', 'RS232-T1200', 'RS232-T1300', 'RS232-T1400', 'RS232-T1500', 'RS232-T1600']
#
# files = ['AES-T2200']

rawpath = "../data/dataset/yosys/PIC16F8"
# rawpath90 = "../data/raw_data_90/"





def graphs_with_port_to_files():
    """
    将网表生成的图 序列化 存储
    :return:
    """
    path = "../data/dic/all_trojan_node.json"
    # 获取所有网表的木马门
    with open(path, 'r') as f:
        allTrojanGateName = json.load(f)

    for file in os.listdir(rawpath):
        filename_without_ext, ext = os.path.splitext(file)
        full_path = os.path.join(rawpath, file)
        GraphSavePath = "../data/netlist_graph/surrogate/" + filename_without_ext + ".pickle"
        # 获取本网表的木马门
        trojanGates = allTrojanGateName[filename_without_ext]
        modlueToGraph = netlistToGraph(full_path, trojanGates)
        with open(GraphSavePath, "wb") as f1:
            pickle.dump(modlueToGraph, f1)

    # 先测试一个文件
    # with open(Graphpath, "rb") as f2:
    #     my_object = pickle.load(f2)
    # print(my_object.graph)
    # print(my_object.moudleName)
    # trojanGateInNetlist = [node for node in graphModule.nodes._nodes if graphModule.nodes._nodes[node]['label'] == 1]
    # GraphAll = list(modlueToGraphs.values())[-1].graph
    # moudleName = list(modlueToGraphs.values())[-1].moudleName


def netlistToGraph(full_path, trojanGates):
    """

    :param fname: 文件名
    :param trojanGates:  文件包含的木马名
    :return: 返回最终的图
    """

    nl_fpath = full_path
    print(full_path)
    modlueToGraphs = {}
    ast, directives = parse([nl_fpath])
    modlueToGraph = None
    # 遍历所有的module
    for module in ast.description.definitions:
        # trojanGates = (str(module.name+trojanGate) for trojanGate in trojanGates)
        moduleParamTuple = module.portlist.ports
        modlueToGraph = mp.ModuleToGraph(module.name)
        graphModule = nx.DiGraph()
        outNetLinkGate = {}
        inputNode = {}
        outputNode = {}
        waitBeConneted = {}
        if isinstance(module, ModuleDef):
            # 遍历module中的所有item
            for instanceListItem in module.items:
                # 处理module的主输入的和主输出
                if isinstance(instanceListItem, Decl):
                    for inputOrOutput in instanceListItem.list:
                        if isinstance(inputOrOutput, Input) or isinstance(inputOrOutput, Output):
                            # graphModule.add_node(inputOrOutput.name)  #添加输入输出的线名为节点名
                            width = inputOrOutput.width
                            if width is not None:
                                lsb = int(width.lsb.value)
                                msb = int(width.msb.value)
                                for i in range(msb - lsb + 1):
                                    strt = module.name+inputOrOutput.name+str(i)
                                    graphModule.add_node(strt)  # 添加输入输出的线名为节点名
                                    if isinstance(inputOrOutput, Input):
                                        graphModule.nodes[strt]['gateType'] = 'inPort'
                                    if isinstance(inputOrOutput, Output):
                                        graphModule.nodes[strt]['gateType'] = 'outPort'
                                    graphModule.nodes[strt]['label'] = 0
                                if isinstance(inputOrOutput, Input):
                                    inputNode[inputOrOutput.name] = tuple(set() for _ in range(msb-lsb+1))
                                if isinstance(inputOrOutput, Output):
                                    # 一般output只有一个gate，但是为了统一，也用set吧
                                    outputNode[inputOrOutput.name] = tuple(set() for _ in range(msb-lsb+1))
                            else:
                                strt = module.name + inputOrOutput.name
                                graphModule.add_node(strt)  # 添加输入输出的线名为节点名
                                graphModule.nodes[strt]['label'] = 0
                                if isinstance(inputOrOutput, Input):
                                    graphModule.nodes[strt]['gateType'] = 'inPort'
                                    inputNode[inputOrOutput.name] = set()
                                if isinstance(inputOrOutput, Output):
                                    graphModule.nodes[strt]['gateType'] = 'outPort'
                                    outputNode[inputOrOutput.name] = set()

                # 处理输出
                if isinstance(instanceListItem, InstanceList):
                    # 遍历所有的Instance，暂时不知道InstanceList是什么样的格式
                    for instance in instanceListItem.instances:
                        gateType = instance.module  # gate类型
                        gateName = module.name+instance.name  # gate名
                        protlist = instance.portlist

                        # 处理gate为module
                        if gateType in modlueToGraphs:

                            modlueToGraphObject = modlueToGraphs[gateType]
                            moduleGateParam = modlueToGraphObject.moduleParamTuple
                            outPortObject = modlueToGraphObject.outputNode  # 获取这个module的输出端口字典
                            """
                            #graphModule has all nodes & edges of both graphs, including attributes
                             Where the attributes conflict, it uses the attributes of graphModule.
                            """


                            # 把gate为module的那张图放进来，好做聚合
                            #
                            copy_graph = modlueToGraphObject.graph.copy()
                            node_mapping = {node: f"{node}{gateName}" for node in modlueToGraphObject.graph.nodes()}
                            nx.relabel_nodes(copy_graph, node_mapping, copy=False)

                            if instance.name in trojanGates:
                                # 设置节点的label属性为1
                                nx.set_node_attributes(copy_graph,
                                                       {n: {'label': 1} for n in copy_graph.nodes()})

                            processed_outPortObject={}

                            #端口节点的命名也要处理
                            for key, value in outPortObject.items():
                                if isinstance(value, tuple):
                                    processed_value=[]
                                    # 处理 tuple
                                    for item in value:
                                        new_set = {str(element) + gateName for element in item}
                                        processed_value.append(new_set)
                                    processed_outPortObject[key]=tuple(processed_value)
                                elif isinstance(value, set):
                                    # 处理 set，转换为 list（如果需要保持顺序）
                                    # 或者直接迭代 set，例如：for item in value:
                                    processed_value = list(str(item) + gateName for item in value)
                                    processed_outPortObject[key] = set(processed_value)

                            graphModule = nx.compose(copy_graph, graphModule)

                            # portDuan.portname为None的情况，就说明是这种格式
                            # RareVectorTrigger TrojanTrigger ( n3279, n2461, Trigger_out);
                            if protlist[0].portname is None:
                                for i in range(len(moduleGateParam)):
                                    portName = moduleGateParam[i].name  # 参数名
                                    if portName in processed_outPortObject:
                                        portLinkNetName = portArgnameIsIdentifierOrPointer(protlist[i].argname)
                                        outNetLinkGate = outNetLinkGateAdd(portLinkNetName, outNetLinkGate,
                                                                           processed_outPortObject[portName])
                                        # 这里删除的应该是这个gatemodule接口的输出端口节点，命名方式为  gateModuleName + TOUT
                                        if graphModule.has_node(gateType+portName):
                                            graphModule.remove_node(gateType+portName)
                            else:
                                # 处理端口
                                for portDuan in protlist:
                                    portType = portDuan.portname  # 端口类型 .SUM
                                    if portType is not None:
                                        # 判断这个端口是不是module的输出端口字典中
                                        if portType in processed_outPortObject:
                                            # 得到这个端口的相连gate集合
                                            # tuple or set
                                            portTuple = processed_outPortObject[portType]
                                            # 这个端口类型下有多个分端口
                                            if isinstance(portTuple, tuple):
                                                j, k, lenj, lsb, msb = (0, 0, 0, 0, 0)
                                                varName = ''

                                                for i in range(len(portTuple)):
                                                    try:
                                                        portLinkNetName, j, k,lenj, lsb, msb, varName = moduleGatePortTuple(
                                                            portDuan.argname, portTuple, i, j, k, lenj, lsb, msb, varName)

                                                        if isinstance(portDuan.argname, Concat):
                                                            if len(portDuan.argname.list) != len(portTuple):
                                                                # Partselect
                                                                # T_142 t2 ( .clk(clk), .in(state[15:8]), .out({p2[7:0], p2[31:8]}) ); p2是本模块的输出端口
                                                                if portLinkNetName in outputNode or varName in outputNode:
                                                                    port_gate_name = module.name + portLinkNetName
                                                                    for ingate in portTuple[i]:
                                                                        if not graphModule.has_edge(ingate, port_gate_name):
                                                                            graphModule.add_edge(ingate, port_gate_name)


                                                                        if isinstance(portDuan.argname.list[k], Partselect):
                                                                            outputNode[varName][lsb - 1].add(ingate)
                                                                        else:
                                                                            outputNode[portLinkNetName].add(ingate)
                                                                        outNetLinkGate = outNetLinkGateAdd(
                                                                            portLinkNetName,
                                                                            outNetLinkGate,
                                                                            ingate)
                                                                else:
                                                                    outNetLinkGate = outNetLinkGateAdd(portLinkNetName,
                                                                                                       outNetLinkGate,
                                                                                                       portTuple[i])

                                                            else:
                                                                port_gate_name = module.name + portLinkNetName
                                                                var = "reset"
                                                                try:
                                                                    if isinstance(portDuan.argname.list[i], Pointer):
                                                                        var = portDuan.argname.list[i].var.name  # PRODUCT
                                                                        ptr = portDuan.argname.list[i].ptr.value  # 25
                                                                except IndexError:
                                                                    print(gateType+"--"+gateName+"--")
                                                                if portLinkNetName in outputNode or var in outputNode:
                                                                    for ingate in portTuple[i]:
                                                                        if not graphModule.has_edge(ingate, port_gate_name):
                                                                            graphModule.add_edge(ingate, port_gate_name)
                                                                        if isinstance(portDuan.argname.list[i], Pointer):
                                                                            outputNode[var][int(ptr)].add(ingate)
                                                                        else:
                                                                            outputNode[portLinkNetName].add(ingate)
                                                                        outNetLinkGate = outNetLinkGateAdd(
                                                                            portLinkNetName,
                                                                            outNetLinkGate,
                                                                            ingate)
                                                                else:
                                                                    outNetLinkGate = outNetLinkGateAdd(portLinkNetName,
                                                                                                       outNetLinkGate,
                                                                                                       portTuple[i])
                                                        #   S_0 S_0 ( .clk(clk), .in(in[31:24]), .out(out[31:24]) );   out是本模块的输出端口
                                                        elif isinstance(portDuan.argname, Partselect):
                                                            if varName in outputNode:
                                                                port_gate_name = module.name + portLinkNetName
                                                                for ingate in portTuple[i]:
                                                                    if not graphModule.has_edge(ingate, port_gate_name):
                                                                        graphModule.add_edge(ingate, port_gate_name)

                                                                    outputNode[varName][lsb-1].add(ingate)
                                                                    outNetLinkGate = outNetLinkGateAdd(
                                                                        portLinkNetName,
                                                                        outNetLinkGate,
                                                                        ingate)
                                                            else:
                                                                outNetLinkGate = outNetLinkGateAdd(portLinkNetName,
                                                                                                   outNetLinkGate,
                                                                                                   portTuple[i])

                                                        elif isinstance(portDuan.argname, Identifier):
                                                            # identifyer的情况
                                                            if portDuan.argname.name in outputNode:
                                                                port_gate_name = module.name + portLinkNetName
                                                                for ingate in portTuple[i]:
                                                                    if not graphModule.has_edge(ingate, port_gate_name):
                                                                        graphModule.add_edge(ingate, port_gate_name)

                                                                    outputNode[portDuan.argname.name][i].add(ingate)
                                                                    outNetLinkGate = outNetLinkGateAdd(
                                                                        portLinkNetName,
                                                                        outNetLinkGate,
                                                                        ingate)
                                                            else:
                                                                outNetLinkGate = outNetLinkGateAdd(portLinkNetName,
                                                                                                   outNetLinkGate,
                                                                                                   portTuple[i])
                                                        else:
                                                            print("else")


                                                    except:
                                                        print(gateType + "--" + gateName + "--" +portType)
                                                    # 删除modulegate的输出端口节点，命名方式为 moduleGatename+.A0
                                                    # portType A  i 0
                                                    if graphModule.has_node(gateType+portType + str(i)):
                                                        graphModule.remove_node(gateType+portType + str(i))
                                            elif isinstance(portTuple, set):

                                                if len(portTuple) == 0:
                                                    if graphModule.has_node(gateType + portType):
                                                        graphModule.remove_node(gateType + portType)
                                                    continue
                                                """
                                                Module moduleGate ( .B({n101772 .....}), .OUT(n101772) )
                                                """
                                                portLinkNetName = portArgnameIsIdentifierOrPointer(
                                                    portDuan.argname)
                                                outNetLinkGate = outNetLinkGateAdd(portLinkNetName, outNetLinkGate,
                                                                                   portTuple)
                                                if graphModule.has_node(gateType + portType):
                                                    graphModule.remove_node(gateType + portType)
                        # 处理line 为 normal gate 的输出端口
                        else:
                            for portDuan in protlist:
                                portType = portDuan.portname  # 端口类型
                                # 为输出端口
                                if portType in ['QN', 'Y', 'CO', 'C1', 'Q', 'SO', 'S', 'ZN']:
                                    if portType == 'S' and gateType not in ["FADDX1", "FADDX2"]:
                                        continue
                                    portLinkNetName = portArgnameIsIdentifierOrPointer(portDuan.argname)
                                    # 存在端口为空没有连接线的情况
                                    if portLinkNetName is None:
                                        continue
                                    portGateName = module.name + portLinkNetName
                                    var = ""
                                    if isinstance(portDuan.argname, Pointer):
                                        var = portDuan.argname.var.name  # A
                                        ptr = portDuan.argname.ptr.value  # 30

                                    if portLinkNetName in outputNode or var in outputNode:
                                        if portLinkNetName in outputNode:
                                            outputNode[portLinkNetName].add(gateName)
                                            if not graphModule.has_edge(gateName, portGateName):
                                                graphModule.add_edge(gateName, portGateName)
                                        if var in outputNode:
                                            outputNode[var][int(ptr)].add(gateName)
                                            if not graphModule.has_edge(gateName, portGateName):
                                                graphModule.add_edge(gateName, portGateName)

                                    outNetLinkGate = outNetLinkGateAdd(portLinkNetName, outNetLinkGate, gateName)
                            # 只有正常gate的节点才需要添加到图中
                            graphModule.add_node(gateName)
                            graphModule.nodes[gateName]['gateType'] = gateType
                            if instance.name in trojanGates:
                                graphModule.nodes[gateName]['label'] = 1
                            else:
                                graphModule.nodes[gateName]['label'] = 0

            # 处理module的输入端口
            for instanceListItem in module.items:
                if isinstance(instanceListItem, InstanceList):
                    for instance in instanceListItem.instances:
                        gateType = instance.module  # 逻辑单元类型
                        gateName = module.name+instance.name  # 逻辑单元名
                        protlist = instance.portlist

                        # 处理gate为module
                        if gateType in modlueToGraphs:
                            modlueToGraphObject = modlueToGraphs[gateType]

                            inPortObject = modlueToGraphObject.inputNode  # 获取这个module的输入端口字典

                            processed_inPortObject={}

                            #端口节点的命名也要处理
                            for key, value in inPortObject.items():
                                if isinstance(value, tuple):
                                    processed_value=[]
                                    # 处理 tuple
                                    for item in value:
                                        new_set = {str(element) + gateName for element in item}
                                        processed_value.append(new_set)
                                    processed_inPortObject[key]=tuple(processed_value)
                                elif isinstance(value, set):
                                    # 处理 set，转换为 list（如果需要保持顺序）
                                    # 或者直接迭代 set，例如：for item in value:
                                    processed_value = list(str(item) + gateName for item in value)
                                    processed_inPortObject[key] = set(processed_value)

                            moduleGateParam = modlueToGraphObject.moduleParamTuple
                            if protlist[0].portname is None:
                                for i in range(len(moduleGateParam)):
                                    portName = moduleGateParam[i].name  # moduleGate 原参数名
                                    if portName in processed_inPortObject:
                                        # mouduleGate的本端口名
                                        portLinkNetName = portArgnameIsIdentifierOrPointer(protlist[i].argname)
                                        if portLinkNetName in outNetLinkGate:
                                            for outGate in outNetLinkGate[portLinkNetName]:
                                                for inGate in processed_inPortObject[portName]:  # inGate可能不止一个
                                                    if not graphModule.has_edge(outGate, inGate):
                                                        graphModule.add_edge(outGate, inGate)
                                        # 这里删除的应该是这个module接口的输出端口节点，命名方式为 moduleGateName+.IN0
                                        if graphModule.has_node(gateType+portName):
                                            graphModule.remove_node(gateType+portName)
                            else:
                                # 处理端口
                                for portDuan in protlist:
                                    portType = portDuan.portname  # 端口类型 .A
                                    # 判断这个端口是不是module的输入端口字典中
                                    if portType in processed_inPortObject:
                                        # 得到这个端口的相连gate集合
                                        portTuple = processed_inPortObject[portType]
                                        # 这个端口类型下有多个分端口
                                        if isinstance(portTuple, tuple):
                                            j, k, lenj, lsb, msb = (0, 0, 0, 0, 0)
                                            varName = ''
                                            for i in range(len(portTuple)):
                                                try:
                                                    portLinkNetName, j, k, lenj, lsb, msb, varName = moduleGatePortTuple(
                                                        portDuan.argname, portTuple, i, j, k, lenj, lsb, msb, varName)


                                                    if portLinkNetName in outNetLinkGate:
                                                        # outGate的数量应该为1
                                                        for outGate in outNetLinkGate[portLinkNetName]:
                                                            for inGate in portTuple[i]:  # inGate可能不止一个
                                                                if not graphModule.has_edge(outGate, inGate):
                                                                    graphModule.add_edge(outGate, inGate)

                                                    elif portLinkNetName in ["1'b0", "1'b1", "1'h0", "1'h1"]:
                                                        if not graphModule.has_node(portLinkNetName):
                                                            graphModule.add_node(portLinkNetName)
                                                            if portLinkNetName in ["1'b0", "1'h0"]:
                                                                graphModule.nodes[portLinkNetName]['gateType'] = "1'b0"
                                                            elif portLinkNetName in ["1'b1", "1'h1"]:
                                                                graphModule.nodes[portLinkNetName]['gateType'] = "1'b1"
                                                            graphModule.nodes[portLinkNetName]['label'] = 0

                                                        for inGate in portTuple[i]:  # inGate可能不止一个
                                                            if not graphModule.has_edge(portLinkNetName, inGate):
                                                                graphModule.add_edge(portLinkNetName, inGate)

                                                    else:

                                                        # .A({ A[4],   A[4]为本模块的入端
                                                        port_gate_name = module.name + portLinkNetName
                                                        if isinstance(portDuan.argname, Partselect):

                                                            if varName in inputNode:
                                                                for ingate in portTuple[i]:
                                                                    if not graphModule.has_edge(port_gate_name, ingate):
                                                                        graphModule.add_edge(port_gate_name, ingate)

                                                                    inputNode[varName][lsb-1].add(ingate)

                                                        # S_138 s0 ( in(in),    这里in是输入端口，并且是in[7:0]类型
                                                        elif isinstance(portDuan.argname, Identifier):
                                                            if portDuan.argname.name in inputNode:
                                                                for ingate in portTuple[i]:
                                                                    if not graphModule.has_edge(port_gate_name, ingate):
                                                                        graphModule.add_edge(port_gate_name, ingate)

                                                                    inputNode[portDuan.argname.name][i].add(ingate)

                                                        elif isinstance(portDuan.argname, Concat):
                                                            if len(portDuan.argname.list) != len(portTuple):
                                                                # Partselect
                                                                # T_142 t2 ( .clk(clk), .in(state[15:8]), .out({p2[7:0], p2[31:8]}) ); p2是本模块的输出端口
                                                                if portLinkNetName in inputNode or varName in inputNode:
                                                                    port_gate_name = module.name + portLinkNetName
                                                                    for ingate in portTuple[i]:
                                                                        if not graphModule.has_edge(port_gate_name, ingate):
                                                                            graphModule.add_edge(port_gate_name, ingate)
                                                                        if isinstance(portDuan.argname.list[k], Partselect):
                                                                            inputNode[varName][lsb - 1].add(ingate)
                                                                        else:
                                                                            inputNode[portLinkNetName].add(ingate)

                                                            else:
                                                                port_gate_name = module.name + portLinkNetName
                                                                var = "reset"
                                                                try:
                                                                    if isinstance(portDuan.argname.list[i], Pointer):
                                                                        var = portDuan.argname.list[i].var.name  # PRODUCT
                                                                        ptr = portDuan.argname.list[i].ptr.value  # 25
                                                                except IndexError:
                                                                    print(gateType+"--"+gateName+"--")
                                                                if portLinkNetName in inputNode or var in inputNode:
                                                                    for ingate in portTuple[i]:
                                                                        if not graphModule.has_edge(port_gate_name, ingate):
                                                                            graphModule.add_edge(port_gate_name, ingate)
                                                                        if isinstance(portDuan.argname.list[i], Pointer):
                                                                            inputNode[var][int(ptr)].add(ingate)
                                                                        else:
                                                                            inputNode[portLinkNetName].add(ingate)

                                                        else:

                                                            if isinstance(portDuan.argname.list[i], Pointer):
                                                                var = portDuan.argname.list[i].var.name  # A
                                                                ptr = portDuan.argname.list[i].ptr.value  # 4
                                                            if portLinkNetName in inputNode or var in inputNode:
                                                                for ingate in portTuple[i]:
                                                                    if not graphModule.has_edge(port_gate_name, ingate):
                                                                        graphModule.add_edge(port_gate_name, ingate)
                                                                    if isinstance(portDuan.argname.list[i], Pointer):
                                                                        inputNode[var][int(ptr)].add(ingate)
                                                                    else:
                                                                        inputNode[portLinkNetName].add(ingate)



                                                    # 移除modulegate 的输入端口节点
                                                    if graphModule.has_node(gateType + portType + str(i)):
                                                        graphModule.remove_node(gateType + portType + str(i))
                                                except:
                                                    print(gateType + "--" + gateName + "--")
                                        elif isinstance(portTuple, set):
                                            # 如果模块对应端口的set()没有值，就删除这个端口节点
                                            if len(portTuple) == 0:
                                                if graphModule.has_node(gateType + portType):
                                                    graphModule.remove_node(gateType + portType)
                                                continue
                                            """
                                            Module moduleGate ( .B({n101772 .....}), .CI(n101773), .A(1'b0) )
                                            """
                                            portLinkNetName = portArgnameIsIdentifierOrPointer(
                                                portDuan.argname)
                                            if isinstance(portDuan.argname, IntConst):
                                                if not graphModule.has_node(portLinkNetName):
                                                    graphModule.add_node(portLinkNetName)
                                                    if portLinkNetName in ["1'b0", "1'h0"]:
                                                        graphModule.nodes[portLinkNetName]['gateType'] = "1'b0"
                                                    elif portLinkNetName in ["1'b1", "1'h1"]:
                                                        graphModule.nodes[portLinkNetName]['gateType'] = "1'b1"
                                                    graphModule.nodes[portLinkNetName]['label'] = 0
                                                for gateName in portTuple:
                                                    graphModule.add_edge(portLinkNetName, gateName)
                                            else:
                                                if portLinkNetName in outNetLinkGate:
                                                    # outGate的数量应该为1
                                                    for outGate in outNetLinkGate[portLinkNetName]:
                                                        for inGate in portTuple:  # inGate可能不止一个
                                                            if not graphModule.has_edge(outGate, inGate):
                                                                graphModule.add_edge(outGate, inGate)

                                                # S_144 s0 ( .clk(clk), .in(in), .out({out_31, out_30, out_29, out_28, out_27,
                                                # clk是本模块的输入端口
                                                elif portLinkNetName in inputNode:
                                                    strt = module.name + portLinkNetName
                                                    for inGate in portTuple:  # inGate可能不止一个
                                                        inputNode[portLinkNetName].add(inGate)
                                                        if not graphModule.has_edge(strt, inGate):
                                                            graphModule.add_edge(strt, inGate)

                                                # 移除modulegate 的输入端口节点
                                                if graphModule.has_node(gateType + portType):
                                                    graphModule.remove_node(
                                                        gateType + portType)
                        else:
                            for portDuan in protlist:
                                portType = portDuan.portname  # 端口类型
                                # 处理gate输入端口
                                if portType not in ['QN', 'Y', 'CO', 'C1', 'Q', 'SO']:
                                    if portType == 'S' and gateType == "FADDX1":
                                        continue
                                    portLinkNetName = portArgnameIsIdentifierOrPointer(portDuan.argname)
                                    # 存在端口为空没有连接线的情况
                                    if portLinkNetName is None:
                                        continue
                                    # 输入端口给个电位 0 或 1
                                    if isinstance(portDuan.argname, IntConst):
                                        # 一般只有输入端口可能有
                                        if not graphModule.has_node(portLinkNetName):
                                            graphModule.add_node(portLinkNetName)
                                            if portLinkNetName in ["1'b0", "1'h0"]:
                                                graphModule.nodes[portLinkNetName]['gateType'] = "1'b0"
                                            elif portLinkNetName in ["1'b1", "1'h1"]:
                                                graphModule.nodes[portLinkNetName]['gateType'] = "1'b1"
                                            graphModule.nodes[portLinkNetName]['label'] = 0
                                        graphModule.add_edge(portLinkNetName, gateName)
                                        continue
                                    # 处理本模块的主输入端口,这里做了一个假设,本模块的主输入不会作为子模块的参数相连
                                    portGateName = module.name + portLinkNetName
                                    var = None
                                    ptr = None
                                    if isinstance(portDuan.argname, Pointer):
                                        var = portDuan.argname.var.name  # A
                                        ptr = portDuan.argname.ptr.value  # 30

                                    if portLinkNetName in inputNode or portLinkNetName in outputNode:
                                        if portLinkNetName in inputNode:

                                            inputNode[portLinkNetName].add(gateName)
                                            if not graphModule.has_edge(portGateName, gateName):
                                                graphModule.add_edge(portGateName, gateName)

                                        elif portLinkNetName in outputNode:
                                            # outputNode[portLinkNetName].add(gateName)
                                            for gate in outputNode[portLinkNetName]:
                                                if not graphModule.has_edge(gate, gateName):
                                                    graphModule.add_edge(gate, gateName)
                                        continue
                                    elif var in inputNode or var in outputNode:
                                        if var in inputNode:
                                            inputNode[var][int(ptr)].add(gateName)
                                            if not graphModule.has_edge(portGateName, gateName):
                                                graphModule.add_edge(portGateName, gateName)

                                        elif var in outputNode:
                                            outputNode[var][int(ptr)].add(gateName)
                                            for gate in outputNode[var][int(ptr)]:
                                                if not graphModule.has_edge(gate, gateName):
                                                    graphModule.add_edge(gate, gateName)
                                        continue

                                    if portLinkNetName in outNetLinkGate:
                                        for outGate in outNetLinkGate[portLinkNetName]:    # outGate的数量应该为1
                                            if not graphModule.has_edge(outGate, gateName):
                                                graphModule.add_edge(outGate, gateName)
                                    else:
                                        #   NAND2X1 U289 ( .A(1'b1), .B(\test_point/TM ), .Y(n370) );
                                        #   \test_point/TM 线不在 outgate中

                                        if portLinkNetName not in waitBeConneted:
                                            waitBeConneted[portLinkNetName] = set()
                                        waitBeConneted[portLinkNetName].add(gateName)

            # 处理module的assign
            # 这里其实是可以优化一下，前面遍历的时候，把assign的语句存起来
            middle = 0
            for instanceListItem in module.items:
                if isinstance(instanceListItem, Assign):
                    left = portArgnameIsIdentifierOrPointer(instanceListItem.left.var)
                    right = portArgnameIsIdentifierOrPointer(instanceListItem.right.var)
                    leftVar = None
                    leftPtr = None
                    rightVar = None
                    rightPtr = None
                    if isinstance(instanceListItem.left.var, Pointer):
                        leftVar = instanceListItem.left.var.var.name  # A
                        leftPtr = instanceListItem.left.var.ptr.value  # 30
                    if isinstance(instanceListItem.right.var, Pointer):
                        rightVar = instanceListItem.right.var.var.name  # A
                        rightPtr = instanceListItem.right.var.ptr.value  # 30
                    # assign wb_dat_o[31] = 1'b0;
                    # assign的左边一般为孤点输出端口
                    if left in outputNode or leftVar in outputNode:
                        leftPortGateName = module.name + left

                        # 右边是 1'b0 1'b1
                        if isinstance(instanceListItem.right.var, IntConst):
                            if isinstance(instanceListItem.left.var, Pointer):
                                outputNode[leftVar][int(leftPtr)].add(right)
                            else:
                                outputNode[left].add(right)
                            if not graphModule.has_node(right):
                                graphModule.add_node(right)
                                if right in ["1'b0", "1'h0"]:
                                    graphModule.nodes[right]['gateType'] = "1'b0"
                                elif right in ["1'b1", "1'h1"]:
                                    graphModule.nodes[right]['gateType'] = "1'b1"
                                graphModule.nodes[right]['label'] = 0
                            if not graphModule.has_edge(right, leftPortGateName):
                                graphModule.add_edge(right, leftPortGateName)

                        # assign g34839 = carry[0];
                        # assign g34839 = g34956;
                        # g34956, carry0 均在 outNetLinkGate 中
                        if right in outNetLinkGate:
                            for outGate in outNetLinkGate[right]:
                                # assign wb_dat_o[31] = g34956;
                                if isinstance(instanceListItem.left.var, Pointer):
                                    outputNode[leftVar][int(leftPtr)].add(outGate)
                                else:
                                    # assign g34839 = g34956;
                                    outputNode[left].add(outGate)
                                if not graphModule.has_edge(outGate, leftPortGateName):
                                    graphModule.add_edge(outGate, leftPortGateName)

                        # assign SUM[0] = A[0] ;
                        if right in inputNode or rightVar in inputNode:
                            middle = middle + 1
                            middleName = module.name+"middle"+str(middle)
                            graphModule.add_node(middleName)
                            graphModule.nodes[middleName]['gateType'] = "middle"
                            graphModule.nodes[middleName]['label'] = 0

                            rightPortGateName = module.name + right
                            if not graphModule.has_edge(middleName, leftPortGateName):
                                graphModule.add_edge(middleName, leftPortGateName)
                            if not graphModule.has_edge(rightPortGateName, middleName):
                                graphModule.add_edge(rightPortGateName, middleName)

                            if isinstance(instanceListItem.right.var, Pointer):
                                inputNode[rightVar][int(rightPtr)].add(middleName)
                            else:
                                inputNode[right].add(middleName)

                            if isinstance(instanceListItem.left.var, Pointer):
                                outputNode[leftVar][int(leftPtr)].add(middleName)
                            else:
                                outputNode[left].add(middleName)

                    else:
                        #  assign \test_point/TM  = test_mode;
                        # 右边为模块的主输入
                        if right in inputNode or rightVar in inputNode:
                            rightPortGateName = module.name + right
                            if left in waitBeConneted:
                                nodesWaitBeConneted = waitBeConneted[left]
                                for node in nodesWaitBeConneted:
                                    if not graphModule.has_edge(rightPortGateName, node):
                                        graphModule.add_edge(rightPortGateName, node)
                                    if isinstance(instanceListItem.right.var, Pointer):
                                        inputNode[rightVar][int(rightPtr)].add(node)
                                    else:
                                        inputNode[right].add(node)


        modlueToGraph.inputNode = inputNode
        modlueToGraph.outputNode = outputNode
        modlueToGraph.graph = graphModule
        modlueToGraph.moduleParamTuple = moduleParamTuple
        modlueToGraphs[modlueToGraph.moudleName] = modlueToGraph

    return modlueToGraph


def moduleGatePortTuple(argnameObject, portTuple, i, j, k, lenj, lsb, msb, varName):
    """
    :param argnameObject:
    :param portTuple: 模型原inputnode[端口类型]  或者 outnode[端口类型]
    :param i: for 循环参数 [0,len(portTuple)]
    :param j: 本moduleGate中对应端口类型的长度
    :param lenj:
    :param lsb:
    :param msb:
    :param varName:
    :return:
    """
    # 端口argname为Concat对象的情况
    k = j
    if isinstance(argnameObject, Concat):
        portDuanArgList = argnameObject.list
        if len(portDuanArgList) == len(portTuple):
            """
            Module moduleGate ( .B({n101773,n101771,n101772 .....}) , .A(...) )
            """
            portLinkNetName = portArgnameIsIdentifierOrPointer(
                portDuanArgList[i])
        else:
            """
            Module moduleGate ( .A({counter[7:6], n101771, n101772, counter[3:0]}), .B({n101773, .....}) )
            """
            if isinstance(portDuanArgList[j],
                          Partselect) and lenj == 0:

                lsb = int(portArgnameIsIdentifierOrPointer(
                    portDuanArgList[j].lsb))
                msb = int(portArgnameIsIdentifierOrPointer(
                    portDuanArgList[j].msb))
                lenj = msb - lsb
                varName = portArgnameIsIdentifierOrPointer(
                    portDuanArgList[j].var)
                portLinkNetName = varName + str(lsb)
                lsb += 1
            elif isinstance(portDuanArgList[j],
                            Partselect) and lenj != 0:
                portLinkNetName = varName + str(lsb)

                lenj -= 1
                if lsb != msb:
                    lsb += 1
                else:
                    lsb += 1
                    j += 1
            else:
                portLinkNetName = portArgnameIsIdentifierOrPointer(
                    portDuanArgList[j])
                j += 1

    elif isinstance(argnameObject, Partselect):
        if isinstance(argnameObject, Partselect) and lenj == 0:

            lsb = int(portArgnameIsIdentifierOrPointer(
                argnameObject.lsb))
            msb = int(portArgnameIsIdentifierOrPointer(
                argnameObject.msb))
            lenj = msb - lsb
            varName = portArgnameIsIdentifierOrPointer(
                argnameObject.var)
            portLinkNetName = varName + str(lsb)
            lsb += 1
        elif isinstance(argnameObject,
                        Partselect) and lenj != 0:
            portLinkNetName = varName + str(lsb)
            lenj -= 1
            if lsb != msb:
                lsb += 1
            else:
                j += 1
                lsb += 1 # 这里也加1，确保lsb输出的都会比本轮大1

    else:
        """
        Module moduleGate ( .A(counter), .B({n101773, .....}) )
        """
        portLinkNetName = argnameObject.name + str(i)

    return portLinkNetName, j, k, lenj, lsb, msb, varName


def portArgnameIsIdentifierOrPointer(argObject):
    portLinkNetName = None
    if isinstance(argObject, Identifier):
        portLinkNetName = argObject.name  # 端口连接的线名
    if isinstance(argObject, Pointer):
        var = argObject.var.name  # A
        ptr = argObject.ptr.value  # 30
        portLinkNetName = var + ptr  # A30
    if isinstance(argObject, IntConst):
        portLinkNetName = argObject.value  # 1'b1, 1'b0
    return portLinkNetName


def outNetLinkGateAdd(portLinkNetName, outNetLinkGate, addObject):
    if portLinkNetName is not None:  # 存在端口为空没有连接线的情况
        if portLinkNetName not in outNetLinkGate:
            outNetLinkGate[portLinkNetName] = set()
        if isinstance(addObject, set):
            outNetLinkGate[portLinkNetName].update(addObject)
        if isinstance(addObject, str):
            outNetLinkGate[portLinkNetName].add(addObject)
    return outNetLinkGate



if __name__ == "__main__":
    graphs_with_port_to_files()
