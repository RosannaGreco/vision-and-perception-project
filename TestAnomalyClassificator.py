#This file allows us to test the anomaly classificator. It will plot an image, the ground truth and the corresponding prediction


import torch

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from MuraDataset import MuraDataset

import matplotlib.pyplot as plt
import torchvision.transforms as tf
from ResNet import ResNet152



test_set = MuraDataset(csv_file='./HUMEROUS_ANOMALIES/humerous_anomalies_final.csv', transform=transforms.ToTensor(),  img_size=64)
test_loader = DataLoader(dataset=test_set, batch_size=5, shuffle=True)


labels_map = {
    0: "fracture",
    1: "screws and plates",
    2: "other anomalies"
}


def GreatestNumberIndex(tensor):
    greatest = -20
    i = 0

    for i in range(0,3):
        if (tensor[i] > greatest):
            greatest = tensor[i]
            greatest_idx = i
        i+=1
    return greatest_idx



model = ResNet152(3, 3)
model.load_state_dict(torch.load('./Models/RESNET_ANOMALIES_CLASSIFICATION.pt'))

for batch_idx, (data, targets) in enumerate(test_loader):
    if (batch_idx < 1):
        img = tf.ToPILImage()(data[0])



        lab =targets[0]
        lab = torch.flatten(lab)
        ground_truth =labels_map[lab.item()]


        scores = model(data)

        prediction_t = scores[0]

        prediction_ = GreatestNumberIndex(prediction_t)
        prediction = labels_map[prediction_]


        plt.figure(figsize=(32,32))
        plt.imshow(img)
        plt.title(f'Ground truth : {ground_truth} \n  Prediction : {prediction}')
        plt.axis('off')
        plt.show()




exit(0)
