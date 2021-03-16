'''
Author: Roy
Date: 2021-03-14 00:02:10
LastEditTime: 2021-03-16 19:34:15
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


class MonomodalImageDataset(Dataset):
    """
    """

    def __init__(self) -> None:
        super(MonomodalImageDataset, self).__init__()
        self.coco_train_pth = "/home/roy/grounding/data/mscoco/images/train2014"
        self.coco_val_pth = "/home/roy/grounding/data/mscoco/images/val2014"
        self.coco_train_img_names = [os.path.join("/home/roy/grounding/data/mscoco/images/train2014", addr) for addr in os.listdir(self.coco_train_pth)]
        self.coco_val_img_names = [os.path.join("/home/roy/grounding/data/mscoco/images/val2014", addr) for addr in os.listdir(self.coco_val_pth)]
        self.total_img_names = self.coco_train_img_names + self.coco_val_img_names
        random.shuffle(self.total_img_names)

    def __len__(self) -> int:
        return len(self.total_img_names)

    def __getitem__(self, index: int):
        # (3, 224, 224)
        return image_transformation(Image.open(self.total_img_names[index]))


def test():
    img_dataset = MonomodalImageDataset()
    img_dataloader = DataLoader(img_dataset, batch_size=4, sampler=RandomSampler(img_dataset))
    for batch in img_dataloader:
        break


if __name__ == '__main__':
    test()
