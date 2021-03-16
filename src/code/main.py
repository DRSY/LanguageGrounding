'''
Author: Roy
Date: 2021-03-14 00:02:05
LastEditTime: 2021-03-16 22:07:59
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /grounding/src/code/main.py
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_transformers import AdamW, WarmupLinearSchedule

import logging
import os
import pprint
from tqdm import tqdm, trange
from config import parse_args, ModelType2Class
from utils import *
from model import VisionModel, AdapterModel, PretrainedModel, VisionModelWithMLP, TranslationModel

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def train_translation_model(args, vision_model, lang_model, translation_model):
    freeze_param(vision_model, True)
    freeze_param(lang_model, True)
    freeze_param(translation_model, False)
    vision_model.eval()
    lang_model.eval()
    translation_model.train()

    # optimizer for translation module
    optimized_param = translation_model.parameters()
    optimizer = optim.Adam(optimized_param, lr=args.pretrain_trans_lr)

    # Train!


def train_grounding():
    pass


def main(args):
    model_type = args.model_type
    model_name = args.model_name
    device = torch.device(args.device)

    # models for pretraining translation model
    vision_base_model = VisionModel().to(device)
    language_base_model = PretrainedModel(model_type, model_name).to(device)
    translation_model = TranslationModel(
        args.vision_size, args.lang_size, args.latent_size, args.trans_nonlinearity).to(device)

    # pretrain trans
    train_translation_model(
        vision_base_model, language_base_model, translation_model)


if __name__ == '__main__':
    args = parse_args()
    main(args)
