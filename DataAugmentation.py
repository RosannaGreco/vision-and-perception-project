
#This script performs Data augmentation starting from a csv file containing image paths and saves the resulting images in a new folder
#It is possible to apply different combinations of operations assigning the relative flags.



#NOTE: the third classification is characterized by these labels
#1 fracture
#2 screws/plates
#3 dislocation
#4 other

import torch
import torchvision
import torchvision.transforms as tf
from PIL import Image
import glob
import pandas as pd
import os
from skimage import io
import csv

#flags
flipping = 1
crop = 0
perspective = 0
rotation = 1

#transformations
transform1 = tf.RandomPerspective(distortion_scale=0.55, p = 1.0)
transform2 = tf.RandomRotation((10,30), expand=True)
transform3 = tf.CenterCrop(380)

index = 0

#file where we write paths and labels of the new images
csv2 = '/home/rosanna/Scrivania/visiope/project/MURA_project/HUMEROUS_ANOMALIES/train_fractures2.csv'
f2 = open(csv2, 'w')
writer = csv.writer(f2)


#open the first csv and modify images
with open('/home/rosanna/Scrivania/visiope/project/MURA_project/HUMEROUS_ANOMALIES/humerus_anomalies.csv') as f1:
    reader = csv.reader(f1)


    for row in reader:
        image_name = 'da'


        img_path = row[0]
        img_label = row[1]

        img = Image.open(img_path)

        index += 1
        if (crop != 0):
            img = transform3(img)
            image_name += '_cropping'
        if (perspective != 0):
            img = transform1(img)
            image_name += '_perspective'
        if (rotation != 0):
            img = transform2(img)
            image_name += '_rotation'

        if (flipping != 0):
            img = tf.ToTensor()(img)
            img = torch.flip(img, [1])
            img = tf.ToPILImage()(img)
            image_name += '_flipping'
        #save images
        new_img_path = ('./DataAugmentation/'+ image_name + str(index) + '.png')
        img = img.save('./DataAugmentation/'+ image_name + str(index) + '.png')

        #write on the second csv
        row = [new_img_path, img_label]

        writer.writerow(row)



f2.close()
f1.close()

#concatenate csvs
file1 = open('/home/rosanna/Scrivania/visiope/project/MURA_project/HUMEROUS_ANOMALIES/humerus_anomalies_final.csv', "a") #file containing all paths and labels
file2 = open('/home/rosanna/Scrivania/visiope/project/MURA_project/HUMEROUS_ANOMALIES/train_fractures2.csv', "r") #f2

for line in file2:
    file1.write(line)
file1.close()
file2.close()
