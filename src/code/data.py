'''
Author: Roy
Date: 2021-03-14 00:02:10
LastEditTime: 2021-03-15 00:01:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /grounding/src/code/data.py
'''
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms
from PIL import Image

# universal transformation for image domain
image_transformation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])