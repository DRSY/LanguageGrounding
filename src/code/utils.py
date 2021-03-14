'''
Author: your name
Date: 2021-03-14 20:24:49
LastEditTime: 2021-03-14 20:44:03
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /grounding/src/code/utils.py
'''
from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        param = param / 1000000.0
        table.add_row([name, "{:.2f}M".format(param)])
        total_params += param
    print(table)
    print("Total Trainable Params: {:.2f}M".format(total_params))
    return total_params


def freeze_param(model, freeze: bool = True):
    for name, p in model.named_parameters():
        p.requires_grad = not freeze
