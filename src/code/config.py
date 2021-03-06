'''
Author: your name
Date: 2021-03-14 00:02:02
LastEditTime: 2021-04-06 23:54:44
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /grounding/src/code/config.py
'''
from argparse import ArgumentParser
from pytorch_transformers import (
    RobertaModel,
    BertModel,
    DistilBertModel,
)
from transformers import MPNetModel, DebertaModel, ElectraModel
from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer, MPNetTokenizer, DebertaTokenizer, ElectraTokenizer

ModelType2Class = {
    'distilbert': (DistilBertModel, DistilBertTokenizer),
    'bert': (BertModel, BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer),
    'mpnet': (MPNetModel, MPNetTokenizer),
    'deberta': (DebertaModel, DebertaTokenizer),
    'electra': (ElectraModel, ElectraTokenizer)
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
    parser.add_argument('--latent_size', type=int, default=1024)
    parser.add_argument('--vision_size', type=int, default=2048)
    parser.add_argument('--lang_size', type=int, default=768)
    parser.add_argument('--pretrain_trans_bs', type=int, default=16)
    parser.add_argument('--pretrain_trans_epochs', type=int, default=2)
    parser.add_argument('--pretrain_trans_lr', type=float, default=1e-3)
    parser.add_argument('--w', type=float, default=0.5)
    parser.add_argument('--pretrain_grounding_bs', type=int, default=16)
    parser.add_argument('--pretrain_grounding_epochs', type=int, default=3)
    parser.add_argument('--grounding_lr', type=float, default=3e-4)
    parser.add_argument('--eval_step', type=int, default=1500)
    parser.add_argument('--loss_type', type=str,
                        choices=['cross', 'simple', 'bilinear'])
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_save', action='store_true')
    args = parser.parse_args()
    return args
