#This file allows us to test the bone classificator. It will plot an image, the ground truth and the corresponding prediction

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from MuraDataset import MuraDataset
from CNN import CNN
import matplotlib.pyplot as plt
import torchvision.transforms as tf


batch_size = 5

test_set = MuraDataset(csv_file='./MURA-v1.1/valid_classes.csv', transform=transforms.ToTensor(),  img_size=32)
test_loader = DataLoader(dataset=test_set, batch_size=5, shuffle=True)


labels_map = {
    0: "elbow",
    1: "finger",
    2: "forearm",
    3: "hand",
    4: "humerus",
    5: "shoulder",
    6: "wrist",
}


def GreatestNumberIndex(tensor):
    greatest = -20
    i = 0

    for i in range(0,7):
        if (tensor[i] > greatest):
            greatest = tensor[i]
            greatest_idx = i
        i+=1
    return greatest_idx



model = CNN(7)
model.load_state_dict(torch.load('./Models/CNN.pt'))

for batch_idx, (data, targets) in enumerate(test_loader):
    if (batch_idx < 1):
        img = tf.ToPILImage()(data[0])



        lab =targets[0]
        lab = torch.flatten(lab)
        ground_truth =labels_map[lab.item()]


        scores = model(data)

        prediction_t = scores[0]
        prediction = GreatestNumberIndex(prediction_t)
        prediction = labels_map[prediction]

        plt.figure(figsize=(32,32))
        plt.imshow(img)
        plt.title(f'Ground truth : {ground_truth} \n  Prediction : {prediction}')
        plt.axis('off')
        plt.show()





