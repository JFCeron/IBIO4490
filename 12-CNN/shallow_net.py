# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 00:22:36 2019

@author: Bananin
"""

import os
import argparse
import numpy as np
import pdb
# torch deep learning
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

def main():
    # network hyperparameters
    num_epochs = 1
    batch_size = 1000
    val_loss_freq = 100 # batch frequency with which we record losses
    lr = 0.001
    # use GPU if available
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # database subsets and loaders
    train_ds = CelebDataset(csv_file='celeba_10_labels_train.csv',root_dir='data_cant_touch_this')
    valid_ds = CelebDataset(csv_file='celeba_10_labels_val.csv',root_dir='data_cant_touch_this')
    test_ds = CelebDataset(csv_file='celeba_10_labels_test.csv',root_dir='data_cant_touch_this')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=10)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=10)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=10)

    # instanciate ShallowNet
    model = ShallowNet()
    model.to(device)
    # loss function: binary cross-entropy
    loss = nn.BCELoss()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)

    # train ShallowNet
    val_losses = [] # store validation loss every epoch for early stopping
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dl):
            images = images.to(device)
            labels = labels.to(device)
            # Run the forward pass
            outputs = model(images)
            pdb.set_trace()
            batch_loss = loss(outputs, labels))

            # optimize
            optimizer.zero_grad() # reset gradients to 0
            loss.backward()
            optimizer.step()

            # keep us informed about the progress
            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                .format(epoch+1, num_epochs, i+1, len(train_dl), loss.item())

        # epoch processed. Calculate validation loss
        losses = AverageMeter('Loss', ':.4e')
        # avoid calculating gradients in forward pass
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(val_dl):
                images = images.to(device)
                target = target.to(device)
                # compute loss in this mini-batch
                output = model(input)
                loss = criterion(output, target)
                losses.update(loss.item(), input.size(0))

        # TODO : guardar el modelo si la perdida en validacion es minima (early stopping)
        # TODO : implementar scheduler para cambiar learning rate dinamicamente


class CelebDataset(Dataset):
    def __init__(self, csv_file, imgs_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_frame = pd.read_csv(csv_file)
        self.imgs_dir = imgs_dir

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        # image name is on first column of labels_frame
        img_name = os.path.join(self.imgs_dir, self.labels_frame.iloc[idx, 0])
        image = Image.open(img_name).ToTensor()
        # this image's labels
        target = self.labels_frame.iloc[idx, 1:11].as_matrix().astype('float')
        return [image, target]

class ShallowNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # out_channels = number of convolutions to apply on this layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=len(canales), out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.fc1 = nn.Sequential(
            nn.Linear(64*1*37, 256), # 37 comes from frequency resolution, 256 from paper
            nn.Sigmoid()
            )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 2),
            nn.Softmax()
            )
    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        # flatten
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        pdb.set_trace()
        return out

# returns the model at filepath
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

if __name__ == "__main__":
    main()
