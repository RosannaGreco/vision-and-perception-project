#this file contains a class useful to build a dataset. 

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import PIL.ImageOps
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as tf


import numpy as np



from PIL import Image, ImageOps

import glob


class MuraDataset(Dataset):
    def __init__(self, csv_file, transform=None, device=None, img_size = 64):
        self.img_labels = pd.read_csv(csv_file)
        self.img_size = img_size
        self.device = device
        self.transform = transform

    def __len__(self):  
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        image= Image.open(img_path)
    #resizing and conversion to RGB

        image = image.resize([self.img_size, self.img_size])
        image = image.convert('RGB')
    #histogram equalization and invertion
        image = PIL.ImageOps.invert(image)
        image = torchvision.transforms.functional.equalize(image)
    


        label = torch.tensor(int(self.img_labels.iloc[idx, 1])).to(self.device)
        if self.transform:
            image = self.transform(image)

        return (image, label)




