from random import Random
from pytorch_transformers.modeling_bert import BertEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset
from torch.distributions import Distribution, Uniform
import logging
import os
import pprint
from config import parse_args, ModelType2Class
from utils import *
from data import *
from made import MADE
import math

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


class CouplingLayer(nn.Module):
    """
    Implementation of the additive coupling layer from section 3.2 of the NICE
    paper.
    """

    def __init__(self, data_dim, hidden_dim, mask, num_layers=4):
        super().__init__()

        assert data_dim % 2 == 0

        self.mask = mask

        modules = [nn.Linear(data_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for i in range(num_layers - 2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.Linear(hidden_dim, data_dim))

        self.m = nn.Sequential(*modules)

    def forward(self, x, logdet, invert=False):
        if not invert:
            x1, x2 = self.mask * x, (1. - self.mask) * x
            y1, y2 = x1, x2 + (self.m(x1) * (1. - self.mask))
            return y1 + y2, logdet

        # Inverse additive coupling layer
        y1, y2 = self.mask * x, (1. - self.mask) * x
        x1, x2 = y1, y2 - (self.m(y1) * (1. - self.mask))
        return x1 + x2, logdet


class ScalingLayer(nn.Module):
    """
    Implementation of the scaling layer from section 3.3 of the NICE paper.
    """

    def __init__(self, data_dim):
        super().__init__()
        self.log_scale_vector = nn.Parameter(
            torch.randn(1, data_dim, requires_grad=True))

    def forward(self, x, invert=False):
        # log_det_jacobian = torch.sum(self.log_scale_vector)

        if invert:
            return torch.exp(- self.log_scale_vector) * x

        return torch.exp(self.log_scale_vector) * x


class LogisticDistribution(Distribution):
    def __init__(self):
        super().__init__()

    def log_prob(self, x):
        return -(F.softplus(x) + F.softplus(-x))

    def sample(self, size):
        z = Uniform(torch.cuda.FloatTensor(
            [0.]), torch.cuda.FloatTensor([1.])).sample(size).to(torch.device('cuda:3'))
        return torch.log(z) - torch.log(1. - z)


class NICE(nn.Module):
    def __init__(self, data_dim, latent_dim, num_coupling_layers=3):
        super().__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim

        # alternating mask orientations for consecutive coupling layers
        masks = [self._get_mask(data_dim, orientation=(i % 2 == 0))
                 for i in range(num_coupling_layers)]

        self.coupling_layers = nn.ModuleList([CouplingLayer(data_dim=data_dim,
                                                            hidden_dim=latent_dim,
                                                            mask=masks[i], num_layers=4)
                                              for i in range(num_coupling_layers)])

        self.scaling_layer = ScalingLayer(data_dim=data_dim)

        self.prior = LogisticDistribution()

    def forward(self, x, invert=False):
        if not invert:
            z = self.f(x)
            return z

        return self.f_inverse(x)

    def f(self, x):
        z = x
        log_det_jacobian = 0
        for i, coupling_layer in enumerate(self.coupling_layers):
            z, log_det_jacobian = coupling_layer(z, log_det_jacobian)
        z = self.scaling_layer(z)
        return z

    def f_inverse(self, z):
        x = z
        x = self.scaling_layer(x, invert=True)
        for i, coupling_layer in reversed(list(enumerate(self.coupling_layers))):
            x, _ = coupling_layer(x, 0, invert=True)
        return x

    def sample(self, num_samples):
        z = self.prior.sample([num_samples, self.data_dim]).view(
            self.samples, self.data_dim)
        return self.f_inverse(z)

    def _get_mask(self, dim, orientation=True):
        mask = np.zeros(dim)
        mask[::2] = 1.
        if orientation:
            mask = 1. - mask     # flip mask orientation
        mask = torch.tensor(mask)
        mask = mask.to(torch.device('cuda:3'))
        return mask.float()


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
        self.cxt_loss = nn.CrossEntropyLoss()
        self.margin_loss_fct = nn.MarginRankingLoss(margin=0.5)
        self.nll = nn.NLLLoss()
        self.cos = nn.CosineSimilarity(dim=1)

        # down project vision feat and lang feat to the same space
        self.vision_downsize = nn.Linear(self.vision_size, self.latent_size)
        self.lang_downsize = nn.Linear(self.language_size, self.latent_size)

        # bilinear verion
        self.bilinear = nn.Bilinear(
            self.vision_size, self.language_size, 1, bias=False)

        # NICE model
        self.nice = NICE(self.latent_size, self.latent_size,
                         num_coupling_layers=2)

        # input pipe
        # self.input_pipe_vision = nn.Sequential(
        #     nn.Linear(vision_size, latent_size),
        #     # self.non_linearity_class(),
        #     # nn.Linear(vision_size, latent_size)
        # )
        # self.input_pipe_language = nn.Sequential(
        #     nn.Linear(language_size, latent_size),
        #     # self.non_linearity_class(),
        #     # nn.Linear(language_size, latent_size)
        # )
        # # output pipe
        # self.output_pipe_vision = nn.Sequential(
        #     nn.Linear(latent_size, vision_size),
        #     # self.non_linearity_class(),
        #     # nn.Linear(vision_size, vision_size)
        # )
        # self.output_pipe_language = nn.Sequential(
        #     nn.Linear(latent_size, language_size),
        #     # self.non_linearity_class(),
        #     # nn.Linear(language_size, language_size)
        # )
        # # one-direction pipe
        # self.vision2lang = nn.Sequential(
        #     self.input_pipe_vision,
        #     self.output_pipe_language
        # )
        # self.language2vision = nn.Sequential(
        #     self.input_pipe_language,
        #     self.output_pipe_vision
        # )
        # # cyclic pipe
        # self.vision2lang2vision = nn.Sequential(
        #     self.vision2lang,
        #     self.language2vision
        # )
        # self.language2vision2language = nn.Sequential(
        #     self.language2vision,
        #     self.vision2lang
        # )

    def NICEFlow(self, feature, invert=False):
        translated_feat = self.nice(feature, invert=invert)
        return translated_feat

    def MSE(self, vision_feat, lang_feat, trans_vision_feat, trans_lang_feat):
        vision_loss = self.loss_fct(trans_vision_feat, vision_feat)
        lang_loss = self.loss_fct(trans_lang_feat, lang_feat)
        return vision_loss + lang_loss

    def contrastive_forward(self, vision_feat, lang_feat):
        """
        vision_feat: (batch_size, vision_size)
        lang`_feat: (batch_size, lang_size)
        """
        bs = vision_feat.shape[0]
        # bilinear version
        vision_feat = vision_feat.unsqueeze(1).repeat(1, bs, 1)
        lang_feat = lang_feat.unsqueeze(0).repeat(bs, 1, 1)
        cosine_sim_matrix = self.bilinear(vision_feat, lang_feat).squeeze(-1) # (bs, bs)
        _label = torch.tensor(list(range(bs))).to(vision_feat.device)
        simple_loss = .0
        simple_loss += self.cxt_loss(cosine_sim_matrix, _label)
        simple_loss += self.cxt_loss(cosine_sim_matrix.transpose(0, 1), _label)
        return simple_loss

        vision_feat = self.vision_downsize(vision_feat)
        lang_feat = self.lang_downsize(lang_feat)
        # simple loss
        vision_acc = 0
        lang_acc = 0
        _vision_feat = vision_feat.unsqueeze(1).transpose(1, 2)
        _lang_feat = lang_feat.unsqueeze(0).transpose(1, 2)
        cos = nn.CosineSimilarity(dim=1)
        cosine_sim_matrix = cos(_vision_feat, _lang_feat) / 0.1  # (bs, bs)
        assert cosine_sim_matrix.shape[0] == cosine_sim_matrix.shape[
            1] == bs, f"{cosine_sim_matrix.shape}"
        _label = torch.tensor(list(range(bs))).to(vision_feat.device)
        _, vision_top2 = torch.topk(cosine_sim_matrix, k=3, dim=-1)
        for i in range(bs):
            if i in vision_top2[i]:
                vision_acc += 1
        _, lang_top2 = torch.topk(
            cosine_sim_matrix.transpose(0, 1), k=3, dim=-1)
        for i in range(bs):
            if i in lang_top2[i]:
                lang_acc += 1
        vision_acc /= bs
        lang_acc /= bs
        simple_loss = .0
        simple_loss += self.cxt_loss(cosine_sim_matrix, _label)
        simple_loss += self.cxt_loss(cosine_sim_matrix.transpose(0, 1), _label)
        return simple_loss, (vision_acc, lang_acc)

        # NICE version
        sqrt_dim = math.sqrt(self.latent_size)
        translated_vision_feat = self.NICEFlow(lang_feat, invert=True)
        translated_lang_feat = self.NICEFlow(vision_feat)

        assert translated_vision_feat.shape == vision_feat.shape
        assert translated_lang_feat.shape == lang_feat.shape

        # margin loss
        positive_scores_vision = torch.sum(
            vision_feat * translated_vision_feat, dim=-1)  # (bs, )
        negtive_scores_vision = torch.sum(torch.cat([vision_feat[1:], vision_feat[0].unsqueeze(
            0)], dim=0) * translated_vision_feat, dim=-1)  # (bs, )
        vision_margin_loss = self.margin_loss_fct(
            positive_scores_vision, negtive_scores_vision, torch.ones(bs).to(vision_feat.device))
        positive_scores_lang = torch.sum(
            lang_feat * translated_lang_feat, dim=-1)  # (bs, )
        negtive_scores_lang = torch.sum(torch.cat([lang_feat[1:], lang_feat[0].unsqueeze(
            0)], dim=0) * translated_lang_feat, dim=-1)  # (bs, )
        lang_margin_loss = self.margin_loss_fct(
            positive_scores_lang, negtive_scores_lang, torch.ones(bs).to(lang_feat.device))
        return vision_margin_loss + lang_margin_loss

        # cross entropy loss
        sim_matrix_vision = torch.matmul(
            vision_feat, translated_vision_feat.transpose(0, 1)).transpose(0, 1) / sqrt_dim  # (bs, bs)
        _, vision_pred = torch.topk(sim_matrix_vision, k=3, dim=-1)
        _label_vision = torch.tensor(list(range(bs))).to(vision_feat.device)
        vision_p3 = 0
        for i in range(bs):
            vision_p3 += 1 if _label_vision[i].item(
            ) in vision_pred[i].tolist() else 0
        vision_p3 /= bs
        infoNCE_loss_vision = self.cxt_loss(sim_matrix_vision, _label_vision)

        sim_matrix_lang = torch.matmul(
            lang_feat, translated_lang_feat.transpose(0, 1)).transpose(0, 1) / sqrt_dim  # (bs, bs)
        _, lang_pred = torch.topk(sim_matrix_lang, k=3, dim=-1)
        _label_lang = torch.tensor(list(range(bs))).to(vision_feat.device)
        lang_p3 = 0
        for i in range(bs):
            lang_p3 += 1 if _label_lang[i].item() in lang_pred[i].tolist() else 0
        lang_p3 /= bs
        infoNCE_loss_lang = self.cxt_loss(sim_matrix_lang, _label_lang)

        infoNCE_loss_total = infoNCE_loss_vision + infoNCE_loss_lang
        return infoNCE_loss_total, (vision_p3, lang_p3)

    def forward(self, vision_feature=None, language_feature=None):
        conicity_loss = 0.0
        if vision_feature is not None:
            # conicity
            _tmp_lang_feat = self.NICEFlow(
                self.vision_downsize(vision_feature))
            avg_tmp_lang_feat = torch.mean(_tmp_lang_feat, dim=0).unsqueeze(0)
            cosine_sim = self.cos(_tmp_lang_feat, avg_tmp_lang_feat)
            lang_conicity = torch.mean(cosine_sim)
            conicity_loss = conicity_loss + lang_conicity
        if language_feature is not None:
            # conicity
            _tmp_vision_feat = self.NICEFlow(
                self.lang_downsize(language_feature), invert=True)
            avg_tmp_vision_feat = torch.mean(
                _tmp_vision_feat, dim=0).unsqueeze(0)
            cosine_sim = self.cos(_tmp_vision_feat, avg_tmp_vision_feat)
            vision_conicity = torch.mean(cosine_sim)
            conicity_loss = conicity_loss + vision_conicity
        return conicity_loss

    def eval_grounding(self, vision_feat, lang_feat, p=2):
        bs = vision_feat.shape[0]
        vision_acc = 0.0
        lang_acc = 0.0
        vision_feat = self.vision_downsize(vision_feat)
        lang_feat = self.lang_downsize(lang_feat)
        _vision_feat = vision_feat.unsqueeze(1).transpose(1, 2)
        _lang_feat = lang_feat.unsqueeze(0).transpose(1, 2)
        cos = nn.CosineSimilarity(dim=1)
        cosine_sim_matrix = cos(_vision_feat, _lang_feat) / 0.1  # (bs, bs)
        assert cosine_sim_matrix.shape[0] == cosine_sim_matrix.shape[
            1] == bs, f"{cosine_sim_matrix.shape}"
        _, vision_top2 = torch.topk(cosine_sim_matrix, k=p, dim=-1)
        for i in range(bs):
            if i in vision_top2[i]:
                vision_acc += 1
        _, lang_top2 = torch.topk(
            cosine_sim_matrix.transpose(0, 1), k=p, dim=-1)
        for i in range(bs):
            if i in lang_top2[i]:
                lang_acc += 1
        vision_acc /= bs
        lang_acc /= bs
        return vision_acc, lang_acc


class GroundedModel(nn.Module):
    """
    Pretrain LM + Adapter model
    """

    def __init__(self, args):
        super().__init__()
        self.model_type = args.model_type
        self.model_name = args.model_name
        self.pretrained_lm = PretrainedModel(self.model_type, self.model_name)
        self.adapter_model = AdapterModel(args, self.pretrained_lm.config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, subj_special_start_id=None, obj_special_start_id=None):
        outputs = self.pretrained_lm(input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask)
        text_feat = self.adapter_model(
            outputs, attention_mask=attention_mask)  # (batch_size, seq_length, hidden_dim)
        return text_feat, outputs

    def load_from_ckpt(self, adapter_ckpt_path: str):
        print(f"Load checkpoint from {adapter_ckpt_path}")
        state_dict = torch.load(adapter_ckpt_path)
        self.adapter_model.load_state_dict(state_dict)


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
    pass
