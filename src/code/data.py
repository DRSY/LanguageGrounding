'''
Author: Roy
Date: 2021-03-14 00:02:10
LastEditTime: 2021-04-06 13:23:51
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /grounding/src/code/data.py
'''
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from PIL import Image
from myutils import image_transformation
import os
import random
import torch
import json
import transformers


class MonomodalImageDataset(Dataset):

    def __init__(self) -> None:
        super(MonomodalImageDataset, self).__init__()
        self.coco_train_pth = "/home/roy/grounding/data/mscoco/images/train2014"
        self.coco_val_pth = "/home/roy/grounding/data/mscoco/images/val2014"
        self.coco_train_img_names = [os.path.join(
            "/home/roy/grounding/data/mscoco/images/train2014", addr) for addr in os.listdir(self.coco_train_pth)]
        self.coco_val_img_names = [os.path.join(
            "/home/roy/grounding/data/mscoco/images/val2014", addr) for addr in os.listdir(self.coco_val_pth)]
        self.total_img_names = self.coco_train_img_names + self.coco_val_img_names
        random.shuffle(self.total_img_names)
        print("Total number of images", len(self.total_img_names))

    def __len__(self) -> int:
        return len(self.total_img_names)

    def __getitem__(self, index: int):
        # (3, 224, 224)
        while True:
            img = Image.open(self.total_img_names[index])
            if img.mode == 'RGB':
                break
            index = random.randint(0, len(self)-1)
        _img = image_transformation(img)
        return _img


class MonomodalTextDataset(Dataset):

    def __init__(self) -> None:
        super(MonomodalTextDataset, self).__init__()
        self.val_caption_pth = "/home/roy/grounding/data/mscoco/captions/mscoco_nominival.json"
        self.train_caption_pth = "/home/roy/grounding/data/mscoco/captions/mscoco_train.json"
        self.val_caption = json.load(open(self.val_caption_pth))
        self.train_caption = json.load(open(self.train_caption_pth))
        self.val_caption_sents = []
        self.train_caption_sents = []
        self.total_caption_sents = []
        for item in self.val_caption:
            for sent in item['sentf']['mscoco']:
                self.val_caption_sents.append(sent)
        for item in self.train_caption:
            for sent in item['sentf']['mscoco']:
                self.train_caption_sents.append(sent)
        del self.train_caption, self.val_caption
        self.total_caption_sents = self.train_caption_sents + self.val_caption_sents
        random.shuffle(self.total_caption_sents)
        print("Total number of sentences:", len(self.total_caption_sents))

    def __len__(self) -> int:
        return len(self.total_caption_sents)

    def __getitem__(self, index: int):
        return self.total_caption_sents[index]


class MonomodalTextCollator(object):

    def __init__(self, tokenizer, max_length: int) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_length

    def __call__(self, batch):
        _input = self.tokenizer(
            batch, return_tensors='pt', max_length=self.max_len, padding='max_length', truncation=True)
        return _input


class PairedCrossModalTrainDataset(Dataset):

    mscoco_train_path: str = "/home/roy/grounding/data/mscoco/images/train2014"
    mscoco_val_path: str = "/home/roy/grounding/data/mscoco/images/val2014"

    def __init__(self) -> None:
        super(PairedCrossModalTrainDataset, self).__init__()
        self.train_caption_pth = "/home/roy/grounding/data/mscoco/captions/mscoco_train.json"
        self.train_caption = json.load(open(self.train_caption_pth))
        self.img_caption_pairs = []
        for item in self.train_caption:
            img_id = item['img_id']
            img_path = os.path.join(self.mscoco_train_path, img_id+".jpg")
            for sent in item['sentf']['mscoco']:
                self.img_caption_pairs.append((img_path, sent))
                # break
        random.shuffle(self.img_caption_pairs)
        print("Total number of paired image-caption for training:",
              len(self.img_caption_pairs))

    def __len__(self) -> int:
        return len(self.img_caption_pairs)

    def __getitem__(self, index: int):
        while 1:
            img_sent_pair = self.img_caption_pairs[index]
            img_obj = Image.open(img_sent_pair[0])
            if img_obj.mode == 'RGB':
                break
            index = random.randint(0, len(self)-1)
        _img = image_transformation(img_obj)
        return _img, img_sent_pair[1]


class PairedCrossModalValDataset(Dataset):

    mscoco_train_path: str = "/home/roy/grounding/data/mscoco/images/train2014"
    mscoco_val_path: str = "/home/roy/grounding/data/mscoco/images/val2014"

    def __init__(self) -> None:
        super(PairedCrossModalValDataset, self).__init__()
        self.val_caption_pth = "/home/roy/grounding/data/mscoco/captions/mscoco_nominival.json"
        self.val_caption = json.load(open(self.val_caption_pth))
        self.img_caption_pairs = []
        for item in self.val_caption:
            img_id = item['img_id']
            img_path = os.path.join(self.mscoco_val_path, img_id+".jpg")
            for sent in item['sentf']['mscoco']:
                self.img_caption_pairs.append((img_path, sent))
                break
        random.shuffle(self.img_caption_pairs)
        print("Total number of paired image-caption for validation:",
              len(self.img_caption_pairs))

    def __len__(self) -> int:
        return len(self.img_caption_pairs)

    def __getitem__(self, index: int):
        while 1:
            img_sent_pair = self.img_caption_pairs[index]
            img_obj = Image.open(img_sent_pair[0])
            if img_obj.mode == 'RGB':
                break
            index = random.randint(0, len(self)-1)
        _img = image_transformation(img_obj)
        return _img, img_sent_pair[1]


class PairedCrossModalCollator:

    def __init__(self, tokenizer, max_length) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch_list):
        packed_img = torch.stack([pair[0]
                                  for pair in batch_list], dim=0)  # (bs, 3, 224, 224)
        captions = [pair[1] for pair in batch_list]
        captions_dict = self.tokenizer(
            captions, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return packed_img, captions_dict


def test():
    tok = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    text_dataset = MonomodalTextDataset()
    img_dataset = MonomodalImageDataset()
    print(len(text_dataset) // len(img_dataset))
    img_bs = 2
    img_dataloader = DataLoader(
        img_dataset, batch_size=img_bs, sampler=RandomSampler(img_dataset))
    text_dataloader = DataLoader(text_dataset, collate_fn=MonomodalTextCollator(
        tok, max_length=20), batch_size=img_bs * (len(text_dataset)//len(img_dataset)), sampler=RandomSampler(text_dataset))
    for i, img_text_batch in enumerate(zip(img_dataloader, text_dataloader)):
        print(img_text_batch[0].shape)
        print(img_text_batch[1].input_ids.shape)
        break


if __name__ == '__main__':
    test()
