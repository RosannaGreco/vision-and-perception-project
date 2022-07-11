
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
from ResNet import ResNet50
from ResNet import ResNet101
from ResNet import ResNet152

from Train import bone_classification, anomaly_detection, anomaly_classification


#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')







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

#
    model = CNN(n_classes)
    bone_classification(model)
# train the second classification
print('-------------------------------------2nd CLASSIFIER TRAINING: recognizing anomalies-------------------------------------------------')
classifier = 2
train = input('Wanna train the  model? [y/n]   ')
if train == "y":
    n_classes = 2
    #model = Resnet34(in_channels=3, outputs=n_classes )
    model = ResNet152(3, n_classes)
    #model = CNN(n_classes)
    model2 = anomaly_detection(model)


load2 = input('Wanna load a pretrained model? [y/n]')
if load2 == "y":
    n_classes = 2
    model = ResNet152(3, n_classes)
    model.load_state_dict(torch.load('./Models/RESNET_ANOMALIES_DETECTION.pt'))
    model.eval()

    train_set = MuraDataset(csv_file='./MURA-v1.1/train_anomalies.csv', transform=transforms.ToTensor(), device=device,
                            img_size=64)
    test_set = MuraDataset(csv_file='./MURA-v1.1/valid_anomalies.csv', transform=transforms.ToTensor(), device=device,
                           img_size=64)

    train_loader = DataLoader(dataset=train_set, batch_size=48, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=48, shuffle=True)

    print('check accuracy on Training set')
    check_accuracy(train_loader, model, device)

    print('check accuracy on Test set')
    check_accuracy(test_loader, model, device)



    print('-------------------------- MODEL LOADED --------------------------')

print('-------------------------------------3nd CLASSIFIER TRAINING: classifying anomalies-------------------------------------------------')
classifier = 2
train = input('Wanna train the  model? [y/n]   ')
if train == "y":
    n_classes = 4
    #model = Resnet34(in_channels=3, outputs=n_classes )
    model = ResNet152(3, n_classes)
    #model = CNN(n_classes)
    model2 = anomaly_classification(model)
load2 = input('Wanna load a pretrained model? [y/n]')
if load2 == "y":
    n_classes = 2
    model = ResNet152(3, n_classes)
    model.load_state_dict(torch.load('./Models/RESNET_ANOMALIES_CLASSIFICATION.pt'))
    model.eval()
    print('-------------------------- MODEL LOADED --------------------------')




exit(0)
