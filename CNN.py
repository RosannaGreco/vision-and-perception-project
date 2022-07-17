#This file defines a simple Convolutional Neural Network

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding = 1)
        self.pool = nn.MaxPool2d(2, 1)
        self.conv2 = nn.Conv2d(6, 32, 5, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 5, padding = 1)
        self.fc1 = nn.Linear(64 * 23 * 23, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)




    def forward(self, x):
        x =self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x =self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)


        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
