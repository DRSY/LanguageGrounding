from argparse import ArgumentParser
from pytorch_transformers import (RobertaTokenizer,
                                  RobertaModel,
                                  BertModel,
                                  BertTokenizer)
ModelType2Class = {
    'roberta': (RobertaModel, RobertaTokenizer),
    'bert': (BertModel, BertTokenizer)
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
    args = parser.parse_args()
    return args
