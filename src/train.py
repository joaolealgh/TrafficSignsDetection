import os
import torch
from datetime import datetime
import torch.nn as nn
import torch.optim as optim

from dataset_functions import *

def test(net, device, test_dataset_loader):
    net.eval()

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for _, test_data in enumerate(test_dataset_loader):
            inputs = test_data['image']
            labels = test_data['label']

            inputs, labels = inputs.to(device), labels.to(device)

            # calculate outputs by running images through the network
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')


def train_one_epoch(net, device, optimizer, criterion, train_dataset_loader):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_dataset_loader):
        # Every data instance is an input + label pair
        inputs = data['image']
        labels = data['label']

        inputs, labels = inputs.to(device), labels.to(device)

        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = net(inputs)

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('batch {} loss: {}'.format(i + 1, last_loss))
            # tb_x = epoch_index * len(train_dataset_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def train_all_epochs(device, net, train_dataset_loader, epochs, validation_dataset_loader, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    best_vloss = 1_000_000.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for epoch in range(epochs):  # loop over the dataset multiple times
        print('EPOCH {}:'.format(epoch + 1))
        running_loss = 0.
        last_loss = 0.

        net.train(True)
        avg_loss = train_one_epoch(net, device, optimizer, criterion, train_dataset_loader)

        running_vloss = 0.0

        net.eval()
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for j, val_data in enumerate(validation_dataset_loader):
                inputs = val_data['image']
                labels = val_data['label']

                inputs, labels = inputs.to(device), labels.to(device)
                
                target = net(inputs)
                vloss = criterion(target, labels)
                running_vloss += vloss
        
            avg_vloss = running_vloss / (j + 1)
            print('LOSS train {} validation {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            # writer.add_scalars('Training vs. Validation Loss',
            #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
            #                 epoch_number + 1)
            # writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'customcnn_{}_{}'.format(timestamp, epoch)
                torch.save(net.state_dict(), 'models/'+model_path)

