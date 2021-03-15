from pytorch_transformers.modeling_bert import BertEncoder
import torch
import torch.nn as nn
import torchvision.models as models

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset
import logging
import os
import pprint
from config import parse_args, ModelType2Class
from utils import *

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class VisionModel(nn.Module):
    def __init__(self):
        super(VisionModel, self).__init__()
        self.full_encoder = models.resnext101_32x8d()

    def forward(self, input):
        pass


class Adapter(nn.Module):
    def __init__(self, args, adapter_config):
        super(Adapter, self).__init__()
        self.adapter_config = adapter_config
        self.args = args
        self.down_project = nn.Linear(
            self.adapter_config.project_hidden_size,
            self.adapter_config.adapter_size,
        )
        self.encoder = BertEncoder(self.adapter_config)
        self.up_project = nn.Linear(
            self.adapter_config.adapter_size, adapter_config.project_hidden_size)
        self.init_weights()

    def forward(self, hidden_states, attention_mask):
        down_projected = self.down_project(hidden_states)

        input_shape = down_projected.size()[:-1]
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=self.args.device)
        encoder_attention_mask = torch.ones(
            input_shape, device=self.args.device)
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        head_mask = [None] * self.adapter_config.num_hidden_layers
        encoder_outputs = self.encoder(down_projected,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask)

        up_projected = self.up_project(encoder_outputs[0])
        return hidden_states + up_projected

    def init_weights(self):
        self.down_project.weight.data.normal_(
            mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(
            mean=0.0, std=self.adapter_config.adapter_initializer_range)
        self.up_project.bias.data.zero_()


class PretrainedModel(nn.Module):
    def __init__(self, model_type, model_name):
        super(PretrainedModel, self).__init__()
        self.model_type = model_type
        self.model_name = model_name
        self.model_class, self.tokenizer_class = ModelType2Class.get(
            model_type)
        self.model = self.model_class.from_pretrained(
            model_name, output_hidden_states=True)
        self.config = self.model.config
        count_parameters(self.model)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, subj_special_start_id=None, obj_special_start_id=None):

        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class AdapterModel(nn.Module):
    def __init__(self, args, pretrained_model_config):
        super(AdapterModel, self).__init__()
        self.config = pretrained_model_config
        self.args = args
        self.adapter_size = self.args.adapter_size

        class AdapterConfig:
            project_hidden_size: int = self.config.hidden_size
            hidden_act: str = "gelu"
            adapter_size: int = self.adapter_size  # 64
            adapter_initializer_range: float = 0.0002
            is_decoder: bool = False
            attention_probs_dropout_prob: float = 0.1
            hidden_dropout_prob: float = 0.1
            hidden_size: int = args.adapter_size
            initializer_range: float = 0.02
            intermediate_size: int = 3072
            layer_norm_eps: float = 1e-05
            max_position_embeddings: int = 514
            num_attention_heads: int = args.adapter_heads
            num_hidden_layers: int = self.args.adapter_transformer_layers
            output_attentions: bool = False
            output_hidden_states: bool = False
            torchscript: bool = False
            type_vocab_size: int = 1
            vocab_size: int = 50265

        self.adapter_skip_layers = self.args.adapter_skip_layers
        # self.config.output_hidden_states=True
        self.adapter_list = [int(i) for i in args.adapter_list.split(",")]
        # self.adapter_list =[int(i) for i in self.adapter_list]
        self.adapter_num = len(self.adapter_list)
        # self.adapter = Adapter(args, AdapterConfig)

        self.adapter = nn.ModuleList(
            [Adapter(args, AdapterConfig) for _ in range(self.adapter_num)])

        self.com_dense = nn.Linear(
            self.config.hidden_size * 2, self.config.hidden_size)

    def forward(self, pretrained_model_outputs, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = pretrained_model_outputs

        # (batch_size, seq_length, project_hidden_size)
        sequence_output = outputs[0]

        # ((batch_size, seq_length, project_hidden_size), ..., ())
        hidden_states = outputs[2]
        if attention_mask is not None:
            assert sequence_output.shape[:-1] == attention_mask.shape
        hidden_states_last = torch.zeros(
            sequence_output.size()).to(self.args.device)

        adapter_hidden_states = []
        adapter_hidden_states_count = 0

        # passing through the adapters
        for i, adapter_module in enumerate(self.adapter):
            fusion_state = hidden_states[self.adapter_list[i]
                                         ] + hidden_states_last
            hidden_states_last = adapter_module(fusion_state, attention_mask)
            adapter_hidden_states.append(hidden_states_last)
            adapter_hidden_states_count += 1
            if self.adapter_skip_layers >= 1:  # if adapter_skip_layers>=1, skip connection
                if adapter_hidden_states_count % self.adapter_skip_layers == 0:
                    hidden_states_last = hidden_states_last + \
                        adapter_hidden_states[int(
                            adapter_hidden_states_count/self.adapter_skip_layers)]

        # final fused representation
        com_features = self.com_dense(
            torch.cat([sequence_output, hidden_states_last], dim=2))
        return com_features

    def save_pretrained(self, save_directory):
        assert os.path.isdir(
            save_directory), "Saving path should be a directory where the model and configuration can be saved"
        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self
        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Saving model checkpoint to %s", save_directory)


def test():
    args = parse_args()
    model_type = args.model_type
    model_name = args.model_name
    device = torch.device(args.device)
    pretrained_model = PretrainedModel(model_type, model_name)
    pretrained_model.to(device)
    pprint.pprint(args)
    adapter_model = AdapterModel(args, pretrained_model.config)
    count_parameters(adapter_model)
    adapter_model.to(device)
    tokenizer = ModelType2Class.get(model_type)[1].from_pretrained(model_name)
    _input = "A man is washing a dish."
    _input_dict = tokenizer(_input, return_tensors='pt').to(device)
    pretrained_model.eval()
    adapter_model.train()
    with torch.no_grad():
        pretrained_model_output = pretrained_model(**_input_dict)
        print(pretrained_model_output[0].shape)
    print(_input_dict.attention_mask)
    combined_output = adapter_model(pretrained_model_output, attention_mask=_input_dict.attention_mask)
    print(combined_output.shape)


if __name__ == '__main__':
    test()
