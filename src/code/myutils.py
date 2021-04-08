'''
Author: your name
Date: 2021-03-14 20:24:49
LastEditTime: 2021-04-02 23:29:27
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /grounding/src/code/utils.py
'''
from prettytable import PrettyTable
import inspect
from torchvision import transforms
import torch
import numpy as np
import random

# universal transformation for image domain
image_transformation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def line_numb():
    '''Returns the current line number in our program'''
    lineno = inspect.currentframe().f_back.f_lineno
    print("current line number: {}".format(lineno))


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def freeze_param(model, freeze: bool = True):
    for name, p in model.named_parameters():
        p.requires_grad = not freeze


def save_model(model, path):
    with open(path, 'wb') as f:
        torch.save(model.state_dict(), f)
