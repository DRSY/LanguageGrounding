'''
Author: Roy
Date: 2021-03-14 00:02:05
LastEditTime: 2021-04-01 11:39:49
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
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
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
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    adapter_param = adapter_model.named_parameters()
    adapter_optimized_param = [
        {'params': [p for n, p in adapter_param if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': args.pretrain_trans_lr},
        {'params': [p for n, p in adapter_param if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.pretrain_trans_lr}
    ]
    optimized_param = list(translation_model.parameters(
    )) + list(vision_model.final_mlp.parameters())
    optimizer = optim.Adam(optimized_param, lr=args.pretrain_trans_lr)
    for group in adapter_optimized_param:
        optimizer.add_param_group(group)

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


@torch.no_grad()
def eval_grounding(args, vision_model, lang_model, adapter_model, translation_model, device, eval_dataloader):
    # eval mode
    vision_model.final_mlp.eval()
    adapter_model.eval()
    translation_model.eval()

    # eval
    logger.info("****** Evaluating ******")
    eval_iterator = tqdm(eval_dataloader, total=len(eval_dataloader))
    vision_accs = []
    lang_accs = []
    for eval_batch in eval_iterator:
        img_input_feat = eval_batch[0].to(device)
        text_input_feat = eval_batch[1].to(device)
        img_feat = vision_model(img_input_feat)
        pretrained_model_output = lang_model(**text_input_feat)
        text_feat = adapter_model(
            pretrained_model_output, attention_mask=text_input_feat.attention_mask)
        text_feat = text_feat[:, 0, :]
        assert len(img_feat.shape) == 2
        assert len(text_feat.shape) == 2
        vision_acc, lang_acc = translation_model.eval_grounding(img_feat, text_feat)
        vision_accs.append(vision_acc)
        lang_accs.append(lang_acc)
    avg_vision_acc = sum(vision_accs) / len(vision_accs)
    avg_lang_acc = sum(lang_accs) / len(lang_accs)

    # restored to train mode
    vision_model.final_mlp.train()
    adapter_model.train()
    translation_model.train()

    return avg_vision_acc, avg_lang_acc


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
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    adapter_param = adapter_model.named_parameters()
    adapter_optimized_param = [
        {'params': [p for n, p in adapter_param if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': args.grounding_lr},
        {'params': [p for n, p in adapter_param if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.grounding_lr}
    ]
    optimized_param = list(translation_model.parameters(
    )) + list(vision_model.final_mlp.parameters())
    optimizer = optim.Adam(optimized_param, lr=args.grounding_lr)
    for group in adapter_optimized_param:
        optimizer.add_param_group(group)

    # Train
    logger.info("****** Contrastive training ******")
    logger.info(f"****** Epochs: {args.pretrain_grounding_epochs} ******")
    logger.info(f"****** vision model: ResNeXt ******")
    logger.info(f"****** language model: {args.model_name} ******")

    # dataset
    tokenizer = ModelType2Class.get(args.model_type)[
        1].from_pretrained(args.model_name)
    paired_dataset_train = PairedCrossModalTrainDataset()
    paired_dataset_val = PairedCrossModalValDataset()
    paired_dataloader_train = DataLoader(paired_dataset_train, batch_size=args.pretrain_grounding_bs, sampler=RandomSampler(
        paired_dataset_train), collate_fn=PairedCrossModalCollator(tokenizer, max_length=20), num_workers=2)
    paired_dataloader_val = DataLoader(paired_dataset_val, batch_size=args.pretrain_grounding_bs, sampler=RandomSampler(
        paired_dataset_val), collate_fn=PairedCrossModalCollator(tokenizer, max_length=20), num_workers=2)

    # Train!
    best_val_acc = .0
    best_vision_acc = .0
    best_lang_acc = .0
    best_epoch = -1
    epoch_iterator = trange(int(args.pretrain_grounding_epochs), desc='Epoch')
    for epoch in epoch_iterator:
        batch_iterator = tqdm(paired_dataloader_train,
                              total=len(paired_dataloader_train))
        for batch_iter, batch in enumerate(batch_iterator):
            img_input_feat = batch[0].to(device)
            text_input_feat = batch[1].to(device)
            img_feat = vision_model(img_input_feat)
            pretrained_model_output = lang_model(**text_input_feat)
            text_feat = adapter_model(
                pretrained_model_output, attention_mask=text_input_feat.attention_mask)
            text_feat = text_feat[:, 0, :]
            assert len(img_feat.shape) == 2
            assert len(text_feat.shape) == 2
            infoNCEloss = translation_model.contrastive_forward(
                img_feat, text_feat)
            optimizer.zero_grad()
            infoNCEloss.backward()
            optimizer.step()
            batch_iterator.set_description(
                "infoNCE loss: {}".format(infoNCEloss.item()))
            if batch_iter % args.eval_step == 0:
                # eval on validation set of MSCOCO
                vision_acc, lang_acc = eval_grounding(args, vision_model, lang_model, adapter_model,
                                                      translation_model, device, paired_dataloader_val)
                logger.info("Grounding Accuracy:")
                logger.info(f"Current: Vision: {vision_acc}, Lang: {lang_acc}")
                if (vision_acc + lang_acc) / 2 > best_val_acc:
                    best_val_acc = (vision_acc + lang_acc) / 2
                    best_vision_acc = vision_acc
                    best_lang_acc = lang_acc
                logger.info(
                    f"Best: Vision: {best_vision_acc}, Lang: {best_lang_acc}")
                save_model(adapter_model, "../../models/adapter.pkl")
                save_model(lang_model, "../../models/language_model.pkl")
                save_model(vision_model, "../../models/ResNeXtMLP.pkl")
                save_model(translation_model,
                           "../../models/translation_model.pkl")
    logger.info(f"Finish. Best val acc: {best_val_acc} epoch: {best_epoch}")


def main(args):
    model_type = args.model_type
    model_name = args.model_name
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # models for pretraining translation model
    vision_base_model = VisionModel().to(device)
    vision_base_model_mlp = VisionModelWithMLP(
        vision_base_model, args.vision_size).to(device)
    language_base_model = PretrainedModel(model_type, model_name).to(device)
    adapter_model = AdapterModel(args, language_base_model.config).to(device)
    translation_model = TranslationModel(
        args.vision_size, args.lang_size, args.latent_size, args.trans_nonlinearity).to(device)

    # pretraining using unpaired image-language data
    # train_translation_model(args,
    #                         vision_base_model_mlp, language_base_model, adapter_model, translation_model, device)

    # pretraining using paired image-caption data from MSCOCO
    train_grounding(args, vision_base_model_mlp, language_base_model,
                    adapter_model, translation_model, device)


if __name__ == '__main__':
    args = parse_args()
    main(args)
