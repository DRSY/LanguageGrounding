'''
Author: Roy
Date: 2021-03-14 00:02:05
LastEditTime: 2021-03-15 13:20:15
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /grounding/src/code/main.py
'''
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_transformers import AdamW, WarmupLinearSchedule

import logging
import os
import pprint
from config import parse_args, ModelType2Class
from utils import *
from model import VisionModel, AdapterModel, PretrainedModel

logger = logging.getLogger(__name__)

def main(args):
    model_type = args.model_type
    model_name = args.model_name
    device = torch.device(args.device)

if __name__ == '__main__':
    args = parse_args()
    main(args)