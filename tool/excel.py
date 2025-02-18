# -*- coding: UTF-8 -*-
'''
@File    ：excel.py
@Author  ：wuqw
@Date    ：2024/11/29 15:30:36
'''


def calculate_column_averages(matrix):
    # 获取二维列表的行数和列数
    num_rows = len(matrix)
    if num_rows == 0:
        return []  # 如果列表为空，返回空列表
    num_cols = len(matrix[0])

    # 初始化一个列表来存储每列的平均值
    column_averages = [0.0] * num_cols

    # 遍历每一列
    for col in range(num_cols):
        column_sum = 0.0
        # 遍历当前列的所有行
        for row in range(num_rows):
            column_sum += matrix[row][col]
        # 计算平均值
        column_averages[col] = column_sum / num_rows

    # 将平均值保留三位小数（这里使用列表推导式进行格式化）
    formatted_averages = ["{:.3f}".format(avg) for avg in column_averages]

    # 如果需要返回浮点数而不是字符串，可以使用round()函数
    # formatted_averages = [round(avg, 3) for avg in column_averages]

    return formatted_averages  # 返回保留三位小数的字符串列表，或者取消注释上一行返回浮点数列表



