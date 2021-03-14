import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

import logging
import os
import pprint
from config import parse_args, ModelType2Class
from utils import count_parameters, freeze_param
from model import VisionModel, AdapterModel, PretrainedModel

logger = logging.getLogger(__name__)

def main(args):
    model_type = args.model_type
    model_name = args.model_name
    device = torch.device(args.device)

if __name__ == '__main__':
    args = parse_args()
    main(args)