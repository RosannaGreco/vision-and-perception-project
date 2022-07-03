
# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from Accuracy import check_accuracy
from torch.utils.data import DataLoader
from MuraDataset import MuraDataset
from torchvision import models
from torchsummary import summary
from CNN import CNN
#from Resnet import Resnet18
#from Resnet import Resnet34
#from Resnet import ResBlock
#from Resnet import Resnet
from Resnet50 import ResNet50
from Resnet50 import ResNet101

from Train import bone_classification, anomaly_classification


#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






# TODO: Build a Resnet model and make a trainable choice
# load possible trained model
print('--------------------------------1st CLASSIFIER TRAINING: recognizing body parts---------------------------------------------------')
classifier = 1
load = input('Load a pretrained model ? [y/n]   ')
if load == "y":
    n_classes = 7
    model = CNN(n_classes)
    model.load_state_dict(torch.load('./Models/CNN.pt'))
    model.eval()
    print('-------------------------- MODEL LOADED --------------------------')
else:
    # train the model from zero
    print('-------------------------- TRAINING THE MODEL --------------------------')
    n_classes = 7

    #model = Resnet18(in_channels = 3 )
# TODO: fix ResNet18 dimensions
    model = CNN(n_classes)
    bone_classification(model)
# TODO: Choose to make a second net or re-train the one already done
# train the second classification
print('-------------------------------------2nd CLASSIFIER TRAINING: recognizing anomalies-------------------------------------------------')
classifier = 2
train = input('Wanna train the  model? [y/n]   ')
if train == "y":
    n_classes = 2
    #model = Resnet34(in_channels=3, outputs=n_classes )
    model = ResNet101(3, n_classes)
    #model = CNN(n_classes)
    model2 = anomaly_classification(model)
load2 = input('Wanna load a pretrained model? [y/n]')
if load2 == "y":
    n_classes = 2
    model = CNN(n_classes)
    model.load_state_dict(torch.load('./Models/CNN_2.pt'))
    model.eval()
    print('-------------------------- MODEL LOADED --------------------------')




exit(0)
