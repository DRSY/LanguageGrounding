'''
Author: your name
Date: 2021-03-14 00:02:02
LastEditTime: 2021-03-15 17:08:29
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /grounding/src/code/config.py
'''
from argparse import ArgumentParser
from pytorch_transformers import (
    RobertaModel,
    BertModel,
    DistilBertModel
)
from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer

ModelType2Class = {
    'roberta': (RobertaModel, RobertaTokenizer),
    'bert': (BertModel, BertTokenizer),
    'distilbert': (DistilBertModel, DistilBertTokenizer)
}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_type', type=str, default='roberta')
    parser.add_argument('--model_name', type=str, default='roberta-large')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--adapter_transformer_layers", default=2, type=int,
                        help="The transformer layers of adapter.")
    parser.add_argument("--adapter_size", default=128, type=int,
                        help="The hidden size of adapter.")
    parser.add_argument("--adapter_heads", default=4, type=int,
                        help="The hidden size of adapter.")
    parser.add_argument("--adapter_list", default="0,6,11", type=str,
                        help="The layer where add an adapter")
    parser.add_argument("--adapter_skip_layers", default=6, type=int,
                        help="The skip_layers of adapter according to bert layers")
    parser.add_argument('--translation_lr', type=float, default=3e-4)
    parser.add_argument('--model_lr', type=float, default=3e-5)
    parser.add_argument('--trans_nonlinearity', type=str,
                        choices=['relu', 'tanh', 'gelu'], default='relu')
    parser.add_argument('--latent_size', type=int, default=3096)
    parser.add_argument('--vision_size', type=int, default=2048)
    parser.add_argument('--lang_size', type=int, default=768)
    args = parser.parse_args()
    return args
