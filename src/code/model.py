from random import Random
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
from data import *

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class VisionModel(nn.Module):
    def __init__(self):
        super(VisionModel, self).__init__()
        self.full_encoder = models.resnext101_32x8d()
        self.backbone_encoder = nn.Sequential(
            *list(self.full_encoder.children())[:-1])

    def forward(self, input):
        output = self.backbone_encoder(input)
        output = output.squeeze(-1).squeeze(-1)
        assert len(output.shape) == 2
        return output


class VisionModelWithMLP(nn.Module):
    def __init__(self, backbone_model, output_size):
        super(VisionModelWithMLP, self).__init__()
        self.output_size = output_size
        self.backbone_model = backbone_model
        self.final_mlp = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        )
        freeze_param(self.backbone_model, freeze=True)

    def forward(self, input):
        o1 = self.backbone_model(input)
        output = self.final_mlp(o1)
        return output


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


class TranslationModel(nn.Module):
    NonLinearity = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'gelu': nn.GELU,
    }

    def __init__(self, vision_size, language_size, latent_size, non_linearity: str):
        super(TranslationModel, self).__init__()
        self.vision_size = vision_size
        self.language_size = language_size
        self.latent_size = latent_size
        self.non_linearity_class = self.NonLinearity.get(
            non_linearity, nn.ReLU)

        self.loss_fct = nn.MSELoss()
        self.nll = nn.NLLLoss()
        self.cos = nn.CosineSimilarity(dim=1)

        self.log_softmax = nn.LogSoftmax(dim=1)

        # input pipe
        self.input_pipe_vision = nn.Sequential(
            nn.Linear(vision_size, vision_size),
            self.non_linearity_class(),
            nn.Linear(vision_size, latent_size)
        )
        self.input_pipe_language = nn.Sequential(
            nn.Linear(language_size, language_size),
            self.non_linearity_class(),
            nn.Linear(language_size, latent_size)
        )
        # output pipe
        self.output_pipe_vision = nn.Sequential(
            nn.Linear(latent_size, vision_size),
            self.non_linearity_class(),
            nn.Linear(vision_size, vision_size)
        )
        self.output_pipe_language = nn.Sequential(
            nn.Linear(latent_size, language_size),
            self.non_linearity_class(),
            nn.Linear(language_size, language_size)
        )
        # one-direction pipe
        self.vision2lang = nn.Sequential(
            self.input_pipe_vision,
            self.output_pipe_language
        )
        self.language2vision = nn.Sequential(
            self.input_pipe_language,
            self.output_pipe_vision
        )
        # cyclic pipe
        self.vision2lang2vision = nn.Sequential(
            self.vision2lang,
            self.language2vision
        )
        self.language2vision2language = nn.Sequential(
            self.language2vision,
            self.vision2lang
        )

    def contrastive_forward(self, vision_feat, lang_feat):
        """
        vision_feat: (batch_size, vision_size)
        lang`_feat: (batch_size, lang_size)
        """
        bs = vision_feat.shape[0]
        translated_vision_feat = self.language2vision(lang_feat)
        translated_lang_feat = self.vision2lang(lang_feat)
        assert translated_vision_feat.shape == vision_feat.shape
        assert translated_lang_feat.shape == lang_feat.shape

        normalized_vision_feat = torch.norm(vision_feat)
        normalized_trans_vision_feat = torch.norm(translated_vision_feat)
        sim_matrix_vision = torch.matmul(
            normalized_vision_feat, normalized_trans_vision_feat)  # (bs, bs)
        logsoftmax_matrix_vision = self.log_softmax(
            sim_matrix_vision)  # (bs, bs)
        _label_vision = torch.tensor(list(range(bs))).to(vision_feat.device)
        infoNCE_loss_vision = self.nll(logsoftmax_matrix_vision, _label_vision)

        normalized_lang_feat = torch.norm(lang_feat)
        normalized_trans_lang_feat = torch.norm(translated_lang_feat)
        sim_matrix_lang = torch.matmul(
            normalized_lang_feat, normalized_trans_lang_feat)  # (bs, bs)
        logsoftmax_matrix_lang = self.log_softmax(sim_matrix_lang)  # (bs, bs)
        _label_lang = torch.tensor(list(range(bs))).to(vision_feat.device)
        infoNCE_loss_lang = self.nll(logsoftmax_matrix_lang, _label_lang)

        infoNCE_loss_total = infoNCE_loss_vision + infoNCE_loss_lang
        return infoNCE_loss_total

    def forward(self, vision_feature=None, language_feature=None):
        cycle_loss = 0.0
        conicity_loss = 0.0
        if vision_feature is not None:
            _tmp_lang_feat = self.vision2lang(vision_feature)
            translated_vision_feature = self.language2vision(_tmp_lang_feat)
            # conicity
            avg_tmp_lang_feat = torch.mean(_tmp_lang_feat, dim=0).unsqueeze(0)
            cosine_sim = self.cos(_tmp_lang_feat, avg_tmp_lang_feat)
            lang_conicity = torch.mean(cosine_sim)
            # cycle consistency
            vision_loss = self.loss_fct(
                translated_vision_feature, vision_feature)
            cycle_loss = cycle_loss + vision_loss
            conicity_loss = conicity_loss + lang_conicity
        if language_feature is not None:
            _tmp_vision_feat = self.language2vision(language_feature)
            translated_lang_feature = self.vision2lang(_tmp_vision_feat)
            # conicity
            avg_tmp_vision_feat = torch.mean(
                _tmp_vision_feat, dim=0).unsqueeze(0)
            cosine_sim = self.cos(_tmp_vision_feat, avg_tmp_vision_feat)
            vision_conicity = torch.mean(cosine_sim)
            # cycle consistency
            lang_loss = self.loss_fct(
                translated_lang_feature, language_feature)
            cycle_loss = cycle_loss + lang_loss
            conicity_loss = conicity_loss + vision_conicity
        return cycle_loss, conicity_loss


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
    combined_output = adapter_model(
        pretrained_model_output, attention_mask=_input_dict.attention_mask)

    translation_model = TranslationModel(2048, 768, 3096, 'relu')
    vision_feature = torch.randn(32, 2048)
    lang_feature = torch.randn(32, 768)
    _translation_loss = translation_model(
        vision_feature=vision_feature, language_feature=lang_feature)
    print(_translation_loss)

    vision_base_model = VisionModel()
    img_dataset = MonomodalImageDataset()
    img_loader = DataLoader(img_dataset, batch_size=16,
                            sampler=RandomSampler(img_dataset))
    for batch in img_loader:
        output = vision_base_model(batch)
        print(output.shape)
        break


if __name__ == '__main__':
    test()
