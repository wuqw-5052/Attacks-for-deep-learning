# 作者：wuqw
# 时间：2023/12/14 9:24

"""
德摩根定律几个替换模式
"""

def get_gate_type_base(gate_type):
    first_index = gate_type.find('X')
    second_index = gate_type.find('X', first_index + 1)
    if second_index == -1:
        # 如果没有找到第二个X，返回第一个X的下标
        index = first_index
    else:
        # 如果找到了第二个X，返回它的下标
        index = second_index
    if gate_type == "MX2X1":
        gate_type_base = "MUX21"
    else:
        if index == -1:
            gate_type_base = gate_type
        else:
            gate_type_base = gate_type[0:index]
    return gate_type_base


# 根据节点类别去调用相对应的变化模式
def lq_classfiy(g, gate):
    gate_type = g.nodes(data=True)[gate]['gateType']
    gate_type_base = get_gate_type_base(gate_type)
    if gate_type_base == 'AND2':
        return rp1(g, gate)
    elif gate_type_base == 'AND3':
        return rp2(g, gate)
    elif gate_type_base == 'AND4':
        return rp3(g, gate)
    elif gate_type_base == 'OR2':
        return rp4(g, gate)
    elif gate_type_base == 'OR3':
        return rp5(g, gate)
    elif gate_type_base == 'OR4':
        return rp6(g, gate)
    elif gate_type_base == 'NAND2':
        return rp7(g, gate)
    elif gate_type_base == 'NAND3':
        return rp8(g, gate)
    elif gate_type_base == 'NAND4':
        return rp9(g, gate)
    elif gate_type_base == 'NOR2':
        return rp10(g, gate)
    elif gate_type_base == 'NOR3':
        return rp11(g, gate)
    elif gate_type_base == 'NOR4':
        return rp12(g, gate)
    elif gate_type_base == 'INV':
        return rp13(g, gate)
    else:
        return common(g, gate)


def rp1(g, nodeTarget):
    """
    rp1是一个2X1 的与门
    :param nodeTarget:
    :return: g
    """
    inputNet = [node for node in g if g.has_edge(node, nodeTarget)]
    outputNet = [node for node in g if g.has_edge(nodeTarget, node)]

    target_lable = g.nodes(data=True)[nodeTarget]['label']

    # 第一个节点 非门
    g.add_node(nodeTarget + str(1))
    g.nodes(data=True)[nodeTarget + str(1)]['gateType'] = "INVX0"
    g.nodes(data=True)[nodeTarget + str(1)]['label'] = target_lable


    # 第二个节点 非门
    g.add_node(nodeTarget + str(2))
    g.nodes(data=True)[nodeTarget + str(2)]['gateType'] = "INVX0"
    g.nodes(data=True)[nodeTarget + str(2)]['label'] = target_lable


    # 第三个节点 两输入或非门
    g.add_node(nodeTarget + str(3))
    g.nodes(data=True)[nodeTarget + str(3)]['gateType'] = "NOR2X1"
    g.nodes(data=True)[nodeTarget + str(3)]['label'] = target_lable


    g.add_edge(nodeTarget + str(1), nodeTarget + str(3))
    g.add_edge(nodeTarget + str(2), nodeTarget + str(3))

    for i in range(len(inputNet)):
        if i == 0:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 1:
            g.add_edge(inputNet[i], nodeTarget + str(2))

    for outNode in outputNet:
        g.add_edge(nodeTarget + str(3), outNode)

    g.remove_node(nodeTarget)
    return g


def rp2(g, nodeTarget):
    """
    rp2是一个3X1 的与门
    :param nodeTarget:
    :return: g
    """
    inputNet = [node for node in g if g.has_edge(node, nodeTarget)]
    outputNet = [node for node in g if g.has_edge(nodeTarget, node)]

    target_lable = g.nodes(data=True)[nodeTarget]['label']

    # 第一个节点 两输入与非门
    g.add_node(nodeTarget + str(1))
    g.nodes(data=True)[nodeTarget + str(1)]['gateType'] = "NAND2X1"
    g.nodes(data=True)[nodeTarget + str(1)]['label'] = target_lable


    # 第二个节点 非门
    g.add_node(nodeTarget + str(2))
    g.nodes(data=True)[nodeTarget + str(2)]['gateType'] = "INVX0"
    g.nodes(data=True)[nodeTarget + str(2)]['label'] = target_lable


    # 第三个节点 两输入或非门
    g.add_node(nodeTarget + str(3))
    g.nodes(data=True)[nodeTarget + str(3)]['gateType'] = "NOR2X1"
    g.nodes(data=True)[nodeTarget + str(3)]['label'] = target_lable

    g.add_edge(nodeTarget + str(1), nodeTarget + str(3))
    g.add_edge(nodeTarget + str(2), nodeTarget + str(3))

    for i in range(len(inputNet)):
        if i == 0:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 1:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 2:
            g.add_edge(inputNet[i], nodeTarget + str(2))

    for outNode in outputNet:
        g.add_edge(nodeTarget + str(3), outNode)

    g.remove_node(nodeTarget)
    return g


def rp3(g, nodeTarget):
    """
    rp3是一个4X1 的与门
    :param nodeTarget:
    :return: g
    """
    inputNet = [node for node in g if g.has_edge(node, nodeTarget)]
    outputNet = [node for node in g if g.has_edge(nodeTarget, node)]

    target_lable = g.nodes(data=True)[nodeTarget]['label']

    # 第一个节点 两输入与非门
    g.add_node(nodeTarget + str(1))
    g.nodes(data=True)[nodeTarget + str(1)]['gateType'] = "NAND2X1"
    g.nodes(data=True)[nodeTarget + str(1)]['label'] = target_lable

    # 第二个节点 两输入与非门
    g.add_node(nodeTarget + str(2))
    g.nodes(data=True)[nodeTarget + str(2)]['gateType'] = "NAND2X1"
    g.nodes(data=True)[nodeTarget + str(2)]['label'] = target_lable

    # 第三个节点 两输入或非门
    g.add_node(nodeTarget + str(3))
    g.nodes(data=True)[nodeTarget + str(3)]['gateType'] = "NOR2X1"
    g.nodes(data=True)[nodeTarget + str(3)]['label'] = target_lable

    g.add_edge(nodeTarget + str(1), nodeTarget + str(3))
    g.add_edge(nodeTarget + str(2), nodeTarget + str(3))

    for i in range(len(inputNet)):
        if i == 0:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 1:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 2:
            g.add_edge(inputNet[i], nodeTarget + str(2))
        elif i == 3:
            g.add_edge(inputNet[i], nodeTarget + str(2))

    for outNode in outputNet:
        g.add_edge(nodeTarget + str(3), outNode)

    g.remove_node(nodeTarget)
    return g


def rp4(g, nodeTarget):
    """
    rp4是一个2X1 的或门
    :param nodeTarget:
    :return: g
    """
    inputNet = [node for node in g if g.has_edge(node, nodeTarget)]
    outputNet = [node for node in g if g.has_edge(nodeTarget, node)]

    target_lable = g.nodes(data=True)[nodeTarget]['label']

    # 第一个节点 非门
    g.add_node(nodeTarget + str(1))
    g.nodes(data=True)[nodeTarget + str(1)]['gateType'] = "INVX0"
    g.nodes(data=True)[nodeTarget + str(1)]['label'] = target_lable

    # 第二个节点 非门
    g.add_node(nodeTarget + str(2))
    g.nodes(data=True)[nodeTarget + str(2)]['gateType'] = "INVX0"
    g.nodes(data=True)[nodeTarget + str(2)]['label'] = target_lable

    # 第三个节点 两输入与非门
    g.add_node(nodeTarget + str(3))
    g.nodes(data=True)[nodeTarget + str(3)]['gateType'] = "NAND2X1"
    g.nodes(data=True)[nodeTarget + str(3)]['label'] = target_lable

    g.add_edge(nodeTarget + str(1), nodeTarget + str(3))
    g.add_edge(nodeTarget + str(2), nodeTarget + str(3))

    for i in range(len(inputNet)):
        if i == 0:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 1:
            g.add_edge(inputNet[i], nodeTarget + str(2))

    for outNode in outputNet:
        g.add_edge(nodeTarget + str(3), outNode)

    g.remove_node(nodeTarget)
    return g


def rp5(g, nodeTarget):
    """
    rp5是一个3X1 的或门
    :param nodeTarget:
    :return: g
    """
    inputNet = [node for node in g if g.has_edge(node, nodeTarget)]
    outputNet = [node for node in g if g.has_edge(nodeTarget, node)]

    target_lable = g.nodes(data=True)[nodeTarget]['label']

    # 第一个节点 两输入与非门
    g.add_node(nodeTarget + str(1))
    g.nodes(data=True)[nodeTarget + str(1)]['gateType'] = "NOR2X1"
    g.nodes(data=True)[nodeTarget + str(1)]['label'] = target_lable

    # 第二个节点 非门
    g.add_node(nodeTarget + str(2))
    g.nodes(data=True)[nodeTarget + str(2)]['gateType'] = "INVX0"
    g.nodes(data=True)[nodeTarget + str(2)]['label'] = target_lable

    # 第三个节点 两输入或非门
    g.add_node(nodeTarget + str(3))
    g.nodes(data=True)[nodeTarget + str(3)]['gateType'] = "NAND2X1"
    g.nodes(data=True)[nodeTarget + str(3)]['label'] = target_lable

    g.add_edge(nodeTarget + str(1), nodeTarget + str(3))
    g.add_edge(nodeTarget + str(2), nodeTarget + str(3))

    for i in range(len(inputNet)):
        if i == 0:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 1:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 2:
            g.add_edge(inputNet[i], nodeTarget + str(2))

    for outNode in outputNet:
        g.add_edge(nodeTarget + str(3), outNode)

    g.remove_node(nodeTarget)
    return g


def rp6(g, nodeTarget):
    """
    rp6是一个4X1 的或门
    :param nodeTarget:
    :return: g
    """
    inputNet = [node for node in g if g.has_edge(node, nodeTarget)]
    outputNet = [node for node in g if g.has_edge(nodeTarget, node)]

    target_lable = g.nodes(data=True)[nodeTarget]['label']

    # 第一个节点 两输入与非门
    g.add_node(nodeTarget + str(1))
    g.nodes(data=True)[nodeTarget + str(1)]['gateType'] = "NOR2X1"
    g.nodes(data=True)[nodeTarget + str(1)]['label'] = target_lable

    # 第二个节点 两输入与非门
    g.add_node(nodeTarget + str(2))
    g.nodes(data=True)[nodeTarget + str(2)]['gateType'] = "NOR2X1"
    g.nodes(data=True)[nodeTarget + str(2)]['label'] = target_lable

    # 第三个节点 两输入或非门
    g.add_node(nodeTarget + str(3))
    g.nodes(data=True)[nodeTarget + str(3)]['gateType'] = "NAND2X1"
    g.nodes(data=True)[nodeTarget + str(3)]['label'] = target_lable

    g.add_edge(nodeTarget + str(1), nodeTarget + str(3))
    g.add_edge(nodeTarget + str(2), nodeTarget + str(3))

    for i in range(len(inputNet)):
        if i == 0:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 1:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 2:
            g.add_edge(inputNet[i], nodeTarget + str(2))
        elif i == 3:
            g.add_edge(inputNet[i], nodeTarget + str(2))

    for outNode in outputNet:
        g.add_edge(nodeTarget + str(3), outNode)

    g.remove_node(nodeTarget)
    return g


def rp7(g, nodeTarget):
    """
    rp7是一个2X1 的与非门
    :param nodeTarget:
    :return: g
    """
    inputNet = [node for node in g if g.has_edge(node, nodeTarget)]
    outputNet = [node for node in g if g.has_edge(nodeTarget, node)]

    target_lable = g.nodes(data=True)[nodeTarget]['label']

    # 第一个节点 非门
    g.add_node(nodeTarget + str(1))
    g.nodes(data=True)[nodeTarget + str(1)]['gateType'] = "INVX0"
    g.nodes(data=True)[nodeTarget + str(1)]['label'] = target_lable

    # 第二个节点 非门
    g.add_node(nodeTarget + str(2))
    g.nodes(data=True)[nodeTarget + str(2)]['gateType'] = "INVX0"
    g.nodes(data=True)[nodeTarget + str(2)]['label'] = target_lable

    # 第三个节点 两输入或非门
    g.add_node(nodeTarget + str(3))
    g.nodes(data=True)[nodeTarget + str(3)]['gateType'] = "OR2X1"
    g.nodes(data=True)[nodeTarget + str(3)]['label'] = target_lable

    g.add_edge(nodeTarget + str(1), nodeTarget + str(3))
    g.add_edge(nodeTarget + str(2), nodeTarget + str(3))

    for i in range(len(inputNet)):
        if i == 0:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 1:
            g.add_edge(inputNet[i], nodeTarget + str(2))

    for outNode in outputNet:
        g.add_edge(nodeTarget + str(3), outNode)

    g.remove_node(nodeTarget)
    return g


def rp8(g, nodeTarget):
    """
    rp8是一个3X1 的与非门
    :param nodeTarget:
    :return: g
    """
    inputNet = [node for node in g if g.has_edge(node, nodeTarget)]
    outputNet = [node for node in g if g.has_edge(nodeTarget, node)]

    target_lable = g.nodes(data=True)[nodeTarget]['label']

    # 第一个节点 两输入与非门
    g.add_node(nodeTarget + str(1))
    g.nodes(data=True)[nodeTarget + str(1)]['gateType'] = "NAND2X1"
    g.nodes(data=True)[nodeTarget + str(1)]['label'] = target_lable

    # 第二个节点 非门
    g.add_node(nodeTarget + str(2))
    g.nodes(data=True)[nodeTarget + str(2)]['gateType'] = "INVX0"
    g.nodes(data=True)[nodeTarget + str(2)]['label'] = target_lable

    # 第三个节点 两输入或非门
    g.add_node(nodeTarget + str(3))
    g.nodes(data=True)[nodeTarget + str(3)]['gateType'] = "OR2X1"
    g.nodes(data=True)[nodeTarget + str(3)]['label'] = target_lable

    g.add_edge(nodeTarget + str(1), nodeTarget + str(3))
    g.add_edge(nodeTarget + str(2), nodeTarget + str(3))

    for i in range(len(inputNet)):
        if i == 0:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 1:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 2:
            g.add_edge(inputNet[i], nodeTarget + str(2))

    for outNode in outputNet:
        g.add_edge(nodeTarget + str(3), outNode)

    g.remove_node(nodeTarget)
    return g


def rp9(g, nodeTarget):
    """
    rp9是一个4X1 的与非门
    :param nodeTarget:
    :return: g
    """
    inputNet = [node for node in g if g.has_edge(node, nodeTarget)]
    outputNet = [node for node in g if g.has_edge(nodeTarget, node)]

    target_lable = g.nodes(data=True)[nodeTarget]['label']

    # 第一个节点 两输入与非门
    g.add_node(nodeTarget + str(1))
    g.nodes(data=True)[nodeTarget + str(1)]['gateType'] = "NAND2X1"
    g.nodes(data=True)[nodeTarget + str(1)]['label'] = target_lable

    # 第二个节点 两输入与非门
    g.add_node(nodeTarget + str(2))
    g.nodes(data=True)[nodeTarget + str(2)]['gateType'] = "NAND2X1"
    g.nodes(data=True)[nodeTarget + str(2)]['label'] = target_lable

    # 第三个节点 两输入或非门
    g.add_node(nodeTarget + str(3))
    g.nodes(data=True)[nodeTarget + str(3)]['gateType'] = "OR2X1"
    g.nodes(data=True)[nodeTarget + str(3)]['label'] = target_lable

    g.add_edge(nodeTarget + str(1), nodeTarget + str(3))
    g.add_edge(nodeTarget + str(2), nodeTarget + str(3))

    for i in range(len(inputNet)):
        if i == 0:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 1:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 2:
            g.add_edge(inputNet[i], nodeTarget + str(2))
        elif i == 3:
            g.add_edge(inputNet[i], nodeTarget + str(2))

    for outNode in outputNet:
        g.add_edge(nodeTarget + str(3), outNode)

    g.remove_node(nodeTarget)
    return g


def rp10(g, nodeTarget):
    """
    rp10是一个2X1 的或非门
    :param nodeTarget:
    :return: g
    """
    inputNet = [node for node in g if g.has_edge(node, nodeTarget)]
    outputNet = [node for node in g if g.has_edge(nodeTarget, node)]

    target_lable = g.nodes(data=True)[nodeTarget]['label']

    # 第一个节点 非门
    g.add_node(nodeTarget + str(1))
    g.nodes(data=True)[nodeTarget + str(1)]['gateType'] = "INVX0"
    g.nodes(data=True)[nodeTarget + str(1)]['label'] = target_lable

    # 第二个节点 非门
    g.add_node(nodeTarget + str(2))
    g.nodes(data=True)[nodeTarget + str(2)]['gateType'] = "INVX0"
    g.nodes(data=True)[nodeTarget + str(2)]['label'] = target_lable

    # 第三个节点 两输入与非门
    g.add_node(nodeTarget + str(3))
    g.nodes(data=True)[nodeTarget + str(3)]['gateType'] = "AND2X1"
    g.nodes(data=True)[nodeTarget + str(3)]['label'] = target_lable

    g.add_edge(nodeTarget + str(1), nodeTarget + str(3))
    g.add_edge(nodeTarget + str(2), nodeTarget + str(3))

    for i in range(len(inputNet)):
        if i == 0:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 1:
            g.add_edge(inputNet[i], nodeTarget + str(2))

    for outNode in outputNet:
        g.add_edge(nodeTarget + str(3), outNode)

    g.remove_node(nodeTarget)
    return g


def rp11(g, nodeTarget):
    """
    rp11是一个3X1 的或非门
    :param nodeTarget:
    :return: g
    """
    inputNet = [node for node in g if g.has_edge(node, nodeTarget)]
    outputNet = [node for node in g if g.has_edge(nodeTarget, node)]

    target_lable = g.nodes(data=True)[nodeTarget]['label']

    # 第一个节点 两输入与非门
    g.add_node(nodeTarget + str(1))
    g.nodes(data=True)[nodeTarget + str(1)]['gateType'] = "NOR2X1"
    g.nodes(data=True)[nodeTarget + str(1)]['label'] = target_lable

    # 第二个节点 非门
    g.add_node(nodeTarget + str(2))
    g.nodes(data=True)[nodeTarget + str(2)]['gateType'] = "INVX0"
    g.nodes(data=True)[nodeTarget + str(2)]['label'] = target_lable

    # 第三个节点 两输入或非门
    g.add_node(nodeTarget + str(3))
    g.nodes(data=True)[nodeTarget + str(3)]['gateType'] = "AND2X1"
    g.nodes(data=True)[nodeTarget + str(3)]['label'] = target_lable

    g.add_edge(nodeTarget + str(1), nodeTarget + str(3))
    g.add_edge(nodeTarget + str(2), nodeTarget + str(3))

    for i in range(len(inputNet)):
        if i == 0:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 1:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 2:
            g.add_edge(inputNet[i], nodeTarget + str(2))

    for outNode in outputNet:
        g.add_edge(nodeTarget + str(3), outNode)

    g.remove_node(nodeTarget)
    return g


def rp12(g, nodeTarget):
    """
    rp12是一个4X1 的或非门
    :param nodeTarget:
    :return: g
    """
    inputNet = [node for node in g if g.has_edge(node, nodeTarget)]
    outputNet = [node for node in g if g.has_edge(nodeTarget, node)]

    target_lable = g.nodes(data=True)[nodeTarget]['label']

    # 第一个节点 两输入与非门
    g.add_node(nodeTarget + str(1))
    g.nodes(data=True)[nodeTarget + str(1)]['gateType'] = "NOR2X1"
    g.nodes(data=True)[nodeTarget + str(1)]['label'] = target_lable

    # 第二个节点 两输入与非门
    g.add_node(nodeTarget + str(2))
    g.nodes(data=True)[nodeTarget + str(2)]['gateType'] = "NOR2X1"
    g.nodes(data=True)[nodeTarget + str(2)]['label'] = target_lable

    # 第三个节点 两输入或非门
    g.add_node(nodeTarget + str(3))
    g.nodes(data=True)[nodeTarget + str(3)]['gateType'] = "AND2X1"
    g.nodes(data=True)[nodeTarget + str(3)]['label'] = target_lable

    g.add_edge(nodeTarget + str(1), nodeTarget + str(3))
    g.add_edge(nodeTarget + str(2), nodeTarget + str(3))

    for i in range(len(inputNet)):
        if i == 0:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 1:
            g.add_edge(inputNet[i], nodeTarget + str(1))
        elif i == 2:
            g.add_edge(inputNet[i], nodeTarget + str(2))
        elif i == 3:
            g.add_edge(inputNet[i], nodeTarget + str(2))

    for outNode in outputNet:
        g.add_edge(nodeTarget + str(3), outNode)

    g.remove_node(nodeTarget)
    return g


def rp13(g, nodeTarget):
    """
        非门的变化模式，输出端口加缓冲器
        :param nodeTarget:
        :return: g
    """

    outputNet = [node for node in g if g.has_edge(nodeTarget, node)]

    target_lable = g.nodes(data=True)[nodeTarget]['label']


    # 缓冲器
    g.add_node(nodeTarget + str(1))
    g.nodes(data=True)[nodeTarget + str(1)]['gateType'] = "BUFX1"
    g.nodes(data=True)[nodeTarget + str(1)]['label'] = target_lable

    g.add_edge(nodeTarget, nodeTarget + str(1))

    for outNode in outputNet:
        g.add_edge(nodeTarget + str(1), outNode)
        g.remove_edge(nodeTarget, outNode)

    return g

def common(g, nodeTarget):
    """
        common 通用的变化模式，输出端口加非门对
        :param nodeTarget:
        :return: g
    """

    outputNet = [node for node in g if g.has_edge(nodeTarget, node)]

    target_lable = g.nodes(data=True)[nodeTarget]['label']


    # 非门对的第一个非门
    g.add_node(nodeTarget + str(1))
    g.nodes(data=True)[nodeTarget + str(1)]['gateType'] = "INVX0"
    g.nodes(data=True)[nodeTarget + str(1)]['label'] = target_lable

    # 非门对的第二个非门
    g.add_node(nodeTarget + str(2))
    g.nodes(data=True)[nodeTarget + str(2)]['gateType'] = "INVX0"
    g.nodes(data=True)[nodeTarget + str(2)]['label'] = target_lable

    g.add_edge(nodeTarget + str(1), nodeTarget + str(2))

    g.add_edge(nodeTarget, nodeTarget + str(1))

    for outNode in outputNet:
        g.add_edge(nodeTarget + str(2), outNode)
        g.remove_edge(nodeTarget, outNode)

    return g
