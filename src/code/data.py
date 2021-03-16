'''
Author: Roy
Date: 2021-03-14 00:02:10
LastEditTime: 2021-03-16 21:57:18
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /grounding/src/code/data.py
'''
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms
from PIL import Image
from utils import image_transformation
import os
import random
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
        return image_transformation(Image.open(self.total_img_names[index]))


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


def test():
    tok = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    text_dataset = MonomodalTextDataset()
    img_dataset = MonomodalImageDataset()
    img_dataloader = DataLoader(
        img_dataset, batch_size=4, sampler=RandomSampler(img_dataset))
    text_dataloader = DataLoader(text_dataset, collate_fn=MonomodalTextCollator(
        tok, max_length=20), batch_size=4, sampler=RandomSampler(text_dataset))
    for tb in text_dataloader:
        print(tb.input_ids.shape)
        break
    for ib in img_dataloader:
        print(ib.shape)
        break


if __name__ == '__main__':
    test()
