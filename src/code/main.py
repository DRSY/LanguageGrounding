'''
Author: Roy
Date: 2021-03-14 00:02:05
LastEditTime: 2021-03-20 14:33:56
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
from tqdm import tqdm, trange
from config import parse_args, ModelType2Class
from utils import *
from model import VisionModel, AdapterModel, PretrainedModel, VisionModelWithMLP, TranslationModel
from data import *

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def train_translation_model(args, vision_model, lang_model, adapter_model, translation_model, device):
    # Trainable modules: [translation model, mlp, adapter model]
    freeze_param(vision_model.backbone_model, True)
    freeze_param(vision_model.final_mlp, False)
    freeze_param(lang_model, True)
    freeze_param(adapter_model, False)
    freeze_param(translation_model, False)
    vision_model.backbone_model.eval()
    vision_model.final_mlp.train()
    lang_model.eval()
    adapter_model.train()
    translation_model.train()

    # optimizers
    optimized_param = list(translation_model.parameters(
    )) + list(vision_model.final_mlp.parameters()) + list(adapter_model.parameters())
    optimizer = optim.Adam(optimized_param, lr=args.pretrain_trans_lr)

    # Train
    logger.info("****** Train translation model ******")
    logger.info(f"****** Epochs: {args.pretrain_trans_epochs} ******")
    logger.info(f"****** vision model: ResNeXt ******")
    logger.info(f"****** language model: {args.model_name} ******")

    # dataset
    tokenizer = ModelType2Class.get(args.model_type)[
        1].from_pretrained(args.model_name)
    text_dataset = MonomodalTextDataset()
    img_dataset = MonomodalImageDataset()
    text_img_ratio = len(text_dataset) // len(img_dataset)

    # dataloader
    img_dataloader = DataLoader(
        img_dataset, batch_size=args.pretrain_trans_bs, sampler=RandomSampler(img_dataset))
    text_dataloader = DataLoader(text_dataset, collate_fn=MonomodalTextCollator(
        tokenizer, max_length=20), batch_size=args.pretrain_trans_bs * text_img_ratio, sampler=RandomSampler(text_dataset))
    epoch_iterator = trange(int(args.pretrain_trans_epochs), desc='Epoch')

    for epoch in epoch_iterator:
        batch_iterator = tqdm(zip(img_dataloader, text_dataloader), total=min(
            [len(img_dataloader), len(text_dataloader)]), desc='Iter')
        for batch_id, img_text_batch in enumerate(batch_iterator):
            img_input_feat, text_input_feat = img_text_batch
            img_input_feat = img_input_feat.to(device)
            text_input_feat = text_input_feat.to(device)
            img_feat = vision_model(img_input_feat)
            pretrained_model_output = lang_model(**text_input_feat)
            text_feat = adapter_model(
                pretrained_model_output, attention_mask=text_input_feat.attention_mask)
            text_feat = text_feat[:, 0, :]
            assert len(img_feat.shape) == 2
            assert len(text_feat.shape) == 2
            cycle_loss, conicity_loss = translation_model(
                vision_feature=img_feat, language_feature=text_feat)
            loss = cycle_loss + args.w * conicity_loss
            assert loss.requires_grad == True
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_iterator.set_description("Loss: {}, Cycle loss: {}, Conicity loss: {}".format(
                loss.item(), cycle_loss.item(), conicity_loss.item()))


def train_grounding(args, vision_model, lang_model, adapter_model, translation_model, device):
    # Trainable modules: [translation model, mlp, adapter model]
    freeze_param(vision_model.backbone_model, True)
    freeze_param(vision_model.final_mlp, False)
    freeze_param(lang_model, True)
    freeze_param(adapter_model, False)
    freeze_param(translation_model, False)
    vision_model.backbone_model.eval()
    vision_model.final_mlp.train()
    lang_model.eval()
    adapter_model.train()
    translation_model.train()

    # optimizers
    optimized_param = list(translation_model.parameters(
    )) + list(vision_model.final_mlp.parameters()) + list(adapter_model.parameters())
    optimizer = optim.Adam(optimized_param, lr=args.pretrain_trans_lr)

    # Train
    logger.info("****** Contrastive training ******")
    logger.info(f"****** Epochs: {args.pretrain_grounding_epochs} ******")
    logger.info(f"****** vision model: ResNeXt ******")
    logger.info(f"****** language model: {args.model_name} ******")


def main(args):
    model_type = args.model_type
    model_name = args.model_name
    device = torch.device(args.device)

    # models for pretraining translation model
    vision_base_model = VisionModel().to(device)
    vision_base_model_mlp = VisionModelWithMLP(
        vision_base_model, args.vision_size).to(device)
    language_base_model = PretrainedModel(model_type, model_name).to(device)
    adapter_model = AdapterModel(args, language_base_model.config).to(device)
    translation_model = TranslationModel(
        args.vision_size, args.lang_size, args.latent_size, args.trans_nonlinearity).to(device)

    # pretrain trans
    train_translation_model(args,
                            vision_base_model_mlp, language_base_model, adapter_model, translation_model, device)

    # pretrain grounding
    train_grounding()


if __name__ == '__main__':
    args = parse_args()
    main(args)
