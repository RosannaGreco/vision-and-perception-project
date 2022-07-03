import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from PIL import Image
import PIL.ImageOps
import torchvision


import numpy as np



from PIL import Image, ImageOps

import glob


class MuraDataset(Dataset):  # initializing images and labels + transforms
    def __init__(self, csv_file, root_dir, transform=None, device=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = root_dir
        self.device = device
        self.transform = transform

    def __len__(self):  # number of samples
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]

        image = io.imread(img_path)
        image= Image.open(img_path)
    #resizing and conversion to RGB
        height, width = (32,32)
        image = image.resize([width, height])
        image = image.convert('RGB')
    #let's equalize and invert the image
        image = PIL.ImageOps.invert(image)
        image = torchvision.transforms.functional.equalize(image)

        label = torch.tensor(int(self.img_labels.iloc[idx, 1])).to(self.device)
        if self.transform:
            image = self.transform(image)

        return (image, label)



#simple codes to check if the images are inverted
#image = Image.open('/home/rosanna/Scrivania/visiope/project/MURA_project/MURA-v1.1/train/XR_FINGER/patient00042/study1_positive/image2.png')

#image = PIL.ImageOps.invert(image)
#image = torchvision.transforms.functional.equalize(image)
#image.show()
