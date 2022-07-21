#this file contains the functions we used to train our models. 

# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from Accuracy import check_accuracy
from torch.utils.data import DataLoader
from MuraDataset import MuraDataset
from CNN import CNN
import matplotlib.pyplot as plt

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
in_channel = 3





def bone_classification(model):
    # data loading
    train_set = MuraDataset(csv_file='./MURA-v1.1/train_classes.csv', transform=transforms.ToTensor(), device=device, img_size= 32)
    test_set = MuraDataset(csv_file='./MURA-v1.1/valid_classes.csv', transform=transforms.ToTensor(), device=device, img_size=32)

    learning_rate = 1e-3
    batch_size = 5
    num_epochs = 10
    
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    # model to cuda
    model = model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    #NEGATIVE LOG LIKELIHOOD
    #m = nn.LogSoftmax(dim = 1)
    #criterion = nn.NLLLoss()


    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)

    # Train Network

    image_losses = []
    for epoch in range(num_epochs):

        losses = []
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):

            # get data to cuda
            data = data.to(device)

            # forward
            scores = model(data)

            #loss = criterion(m(scores), targets)
            loss = criterion(scores, targets)

            losses.append(loss.item())


            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            running_loss = loss.item()

            if batch_idx % 100 == 0:
                print(f'Image progression: {batch_idx}')
                image_loss = running_loss
                image_losses.append(image_loss)

        print(f'Cost at epoch {epoch} is {sum(losses) / len(losses)}')
        
    plt.plot(image_losses)
    plt.show()
    print('check accuracy on Training set')
    check_accuracy(train_loader, model, device)

    print('check accuracy on Test set')
    check_accuracy(test_loader, model, device)

    print('-------------------------- MODEL TRAINED --------------------------')

    save = input('Wanna save the trained model ? [y/n]  ')
    if save == "y":
        torch.save(model.state_dict(), './Models/CNN.pt')

    return model



def anomaly_detection(model):
    # data loading
    batch_size = 48
    num_epochs = 15
    learning_rate= 1e-4
    train_set = MuraDataset(csv_file='./MURA-v1.1/train_anomalies.csv', transform=transforms.ToTensor(), device=device, img_size=64)
    test_set = MuraDataset(csv_file='./MURA-v1.1/valid_anomalies.csv',transform=transforms.ToTensor(), device=device, img_size=64)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    model.to(device)

    # loss and optimizer

    #criterion = nn.CrossEntropyLoss() #to test cross entropy loss
    
    # NEGATIVE LOG LIKELIHOOD
    m = nn.LogSoftmax(dim = 1) #softmax operation
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network
    image_losses = []
    for epoch in range(num_epochs):
        losses = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            # get data to cuda
            data = data.to(device)

            # forward
            scores = model(data)
            loss = criterion(m(scores), targets)
            #loss = criterion(scores, targets) #to test cross entropy loss
            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
            running_loss = loss.item()
            if batch_idx % 100 == 0:
                print(f'Image progression: {batch_idx}')
                image_loss = running_loss
                image_losses.append(image_loss)
        print(f'Cost at epoch {epoch} is {sum(losses) / len(losses)}')
    plt.plot(image_losses)
    plt.show()
    print('check accuracy on Training set')
    check_accuracy(train_loader, model, device)

    print('check accuracy on Test set')
    check_accuracy(test_loader, model, device)

    print('-------------------------- MODEL TRAINED --------------------------')

    save = input('wanna save the trained model ? [y/n]  ')
    if save == "y":
        torch.save(model.state_dict(), './Models/RESNET_ANOMALIES_DETECTION.pt')

    return model

def anomaly_classification(model):
    # data loading

    dataset = MuraDataset(csv_file='./HUMEROUS_ANOMALIES/humerous_anomalies_final.csv', transform=transforms.ToTensor(), device=device, img_size= 64)

    train_set, test_set = torch.utils.data.random_split(dataset, [3375,600])

    learning_rate = 1e-4
    batch_size = 5
    num_epochs = 20
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    model.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Network
    image_losses = []
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
            running_loss = loss.item()
            if batch_idx % 100 == 0:
                print(f'Image progression: {batch_idx}')
                image_loss = running_loss
                image_losses.append(image_loss)
        print(f'Cost at epoch {epoch} is {sum(losses) / len(losses)}')
    plt.plot(image_losses)
    plt.show()
    print('check accuracy on Training set')
    check_accuracy(train_loader, model, device)

    print('check accuracy on Test set')
    check_accuracy(test_loader, model, device)

    print('-------------------------- MODEL TRAINED --------------------------')

    save = input('wanna save the trained model ? [y/n]  ')
    if save == "y":
        torch.save(model.state_dict(), './Models/RESNET_ANOMALIES_CLASSIFICATION.pt')

    return model
