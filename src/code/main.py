'''
Author: Roy
Date: 2021-03-14 00:02:05
LastEditTime: 2021-04-07 19:09:35
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /grounding/src/code/main.py
'''
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_transformers import AdamW, WarmupLinearSchedule

import logging
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm, trange
from typing_extensions import final
from config import parse_args, ModelType2Class
from myutils import *
from model import VisionModel, AdapterModel, PretrainedModel, VisionModelWithMLP, TranslationModel, GroundedModel
from multipleChoice import GroundedModelForMultiplceChoice
from sequenceClassification import GroundedModelForSequenceClassification
from data import *
from transformers import AutoTokenizer

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
        img_dataset, batch_size=args.pretrain_trans_bs, sampler=RandomSampler(img_dataset), num_workers=2)
    text_dataloader = DataLoader(text_dataset, collate_fn=MonomodalTextCollator(
        tokenizer, max_length=20), batch_size=args.pretrain_trans_bs * text_img_ratio, sampler=RandomSampler(text_dataset), num_workers=2)
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
            # text_feat = text_feat[:, 0, :]
            text_feat = torch.mean(text_feat, dim=1)
            assert len(img_feat.shape) == 2
            assert len(text_feat.shape) == 2
            conicity_loss = translation_model(
                vision_feature=img_feat, language_feature=text_feat)
            loss = conicity_loss
            assert loss.requires_grad == True
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_iterator.set_description("Loss: {}, Conicity loss: {}".format(
                loss.item(), conicity_loss.item()))


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
        text_feat = torch.mean(text_feat, dim=1)
        assert len(img_feat.shape) == 2
        assert len(text_feat.shape) == 2
        if args.loss_type == 'cross':
            vision_acc, lang_acc = translation_model.eval_grounding_cross(
                img_feat, text_feat, p=3)
        else:
            vision_acc, lang_acc = translation_model.eval_grounding(
                img_feat, text_feat, p=3)
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
    paired_dataloader_val = DataLoader(paired_dataset_val, batch_size=args.pretrain_grounding_bs,
                                       shuffle=False, collate_fn=PairedCrossModalCollator(tokenizer, max_length=20), num_workers=2)

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
            text_feat = torch.mean(text_feat, dim=1)
            assert len(img_feat.shape) == 2
            assert len(text_feat.shape) == 2
            if args.loss_type == 'cross':
                infoNCEloss_lang = translation_model.NICE_langloss(
                    img_feat, text_feat)
                optimizer.zero_grad()
                infoNCEloss_lang.backward(retain_graph=True)
                optimizer.step()
                infoNCEloss_vision = translation_model.NICE_visionloss(
                    text_feat, img_feat)
                optimizer.zero_grad()
                infoNCEloss_vision.backward()
                optimizer.step()
                infoNCEloss = infoNCEloss_vision + infoNCEloss_lang
            else:
                try:
                    infoNCEloss, (v_acc, l_acc) = translation_model.contrastive_forward(
                        img_feat, text_feat)
                except:
                    infoNCEloss = translation_model.contrastive_forward(
                        img_feat, text_feat)
                finally:
                    pass
                optimizer.zero_grad()
                infoNCEloss.backward()
                optimizer.step()
            try:
                batch_iterator.set_description(
                    "infoNCE loss: {}, vision_acc: {}, lang_acc: {}".format(infoNCEloss.item(), v_acc, l_acc))
            except:
                batch_iterator.set_description(
                    "infoNCE loss: {}".format(infoNCEloss.item()))
            if batch_iter >= 0 and batch_iter % args.eval_step == 0:
                # eval on validation set of MSCOCO
                vision_acc, lang_acc = eval_grounding(args, vision_model, lang_model, adapter_model,
                                                      translation_model, device, paired_dataloader_val)
                logger.info("Grounding Accuracy:")
                logger.info(f"Current: Vision: {vision_acc}, Lang: {lang_acc}")
                if (vision_acc + lang_acc) / 2 > best_val_acc:
                    best_val_acc = (vision_acc + lang_acc) / 2
                    best_vision_acc = vision_acc
                    best_lang_acc = lang_acc
                    best_epoch = epoch
                    # save model checkpoint with highest val acc
                    if args.do_save:
                        save_model(adapter_model,
                                   f"../../models/{args.loss_type}_adapter_{args.model_type}.pkl")
                        save_model(
                            lang_model, f"../../models/{args.loss_type}_{args.model_type}.pkl")
                        save_model(
                            vision_model, f"../../models/{args.loss_type}_ResNeXtMLP_{args.model_type}.pkl")
                        save_model(translation_model,
                                   f"../../models/{args.loss_type}_translation_model_{args.model_type}.pkl")
                logger.info(
                    f"Best: Vision: {best_vision_acc}, Lang: {best_lang_acc}")
    logger.info(f"Finish. Best val acc: {best_val_acc} epoch: {best_epoch}")


def main(args):
    # set random seed
    # set_seed(args.seed)

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


def test(args):
    # multiple-choice
    grounded_mc = GroundedModelForMultiplceChoice(args)
    #grounded_mc.load_adapter_from_ckpt(
    #    f"/home/roy/grounding/models/{args.loss_type}_adapter_{args.model_type}.pkl")
    grounded_mc.to(torch.device('cuda:1'))

    # sanity check
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    prompt = "I love listening to music."
    c1 = "Therefore I am happy everyday."
    c2 = "I hate my life."
    c3 = "I hate my life."
    input_dict = tokenizer([prompt, prompt, prompt], [c1, c2, c3], return_tensors='pt', padding=True).to(torch.device('cuda:1'))
    label = torch.tensor(0).unsqueeze(0).to(torch.device('cuda:1'))
    loss, logits = grounded_mc(**{k: v.unsqueeze(0) for k,v in input_dict.items()}, labels=label)
    print(loss)
    print(logits.shape)

    # # # sequence classification
    # grounded_cls = GroundedModelForSequenceClassification(args, num_classes=2)
    # # grounded_cls.load_adapter_from_ckpt(
    # #     f"/home/roy/grounding/models/{args.loss_type}_adapter_{args.model_type}.pkl")
    # grounded_cls.to(torch.device('cuda:3'))
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # sentence = "I really enjoy the food here."
    # labels = torch.tensor(1).unsqueeze(0).to(torch.device('cuda:3'))
    # input_dict = tokenizer(sentence, return_tensors='pt').to(
    #     torch.device('cuda:3'))
    # loss, logits = grounded_cls(**input_dict, labels=labels)
    # print(loss)
    # print(logits.shape)


if __name__ == '__main__':
    args = parse_args()
    if args.do_train:
        main(args)
    elif args.do_test:
        test(args)
