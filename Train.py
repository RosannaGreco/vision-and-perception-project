# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from Accuracy import check_accuracy
from torch.utils.data import DataLoader
from MuraDataset import MuraDataset
from CNN import CNN

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
in_channel = 3
num_classes = 7
learning_rate = 1e-3
batch_size = 5
num_epochs = 5


def bone_classification(model):
    # data loading
    train_set = MuraDataset(csv_file='./MURA-v1.1/train_classes.csv',
                            root_dir='./MURA-v1.1/train', transform=transforms.ToTensor(), device=device)
    test_set = MuraDataset(csv_file='./MURA-v1.1/valid_classes.csv',
                           root_dir='./MURA-v1.1/valid', transform=transforms.ToTensor(), device=device)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    # model to cuda
    model = model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)

    # Train Network
    for epoch in range(num_epochs):
        losses = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            # get data to cuda
            data = data.to(device)

            # forward
            scores = model(data)

            loss = criterion(scores, targets)

            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Image progression: {batch_idx}')
        print(f'Cost at epoch {epoch} is {sum(losses) / len(losses)}')
    print('check accuracy on Training set')
    check_accuracy(train_loader, model, device)

    print('check accuracy on Test set')
    check_accuracy(test_loader, model, device)

    print('-------------------------- MODEL TRAINED --------------------------')

    save = input('Wanna save the trained model ? [y/n]  ')
    if save == "y":
        torch.save(model.state_dict(), './Models/CNN.pt')

    return model



def anomaly_classification(model):
    # data loading
    train_set = MuraDataset(csv_file='/home/rosanna/Scrivania/visiope/project/MURA_project/MURA-v1.1/train_anomalies.csv',
                            root_dir='/home/rosanna/Scrivania/visiope/project/MURA_project/MURA-v1.1/train', transform=transforms.ToTensor(), device=device)
    test_set = MuraDataset(csv_file='/home/rosanna/Scrivania/visiope/project/MURA_project/MURA-v1.1/valid_anomalies.csv',
                           root_dir='/home/rosanna/Scrivania/visiope/project/MURA_project/MURA-v1.1/valid', transform=transforms.ToTensor(), device=device)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network
    for epoch in range(num_epochs):
        losses = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            # get data to cuda
            data = data.to(device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)
            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Image progression: {batch_idx}')
        print(f'Cost at epoch {epoch} is {sum(losses) / len(losses)}')

    print('check accuracy on Training set')
    check_accuracy(train_loader, model, device)

    print('check accuracy on Test set')
    check_accuracy(test_loader, model, device)

    print('-------------------------- MODEL TRAINED --------------------------')

    save = input('wanna save the trained model ? [y/n]  ')
    if save == "y":
        torch.save(model.state_dict(), './Models/RESNET.pt')

    return model
