"""
Created on Thr Apr 18 08:15:43 2019

@author: BATMAN
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler

from skimage import io, transform
from PIL import Image

import numpy as np
import pandas as pd

class CelebDataset(Dataset):
    """Celeb dataset."""

    def __init__(self, csv_file, root_dir, transform, target_transform):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        """
        Following code from LAB11 and MNIST.
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.open(img_name)
        
        target = self.landmarks_frame.iloc[idx, 1:11].as_matrix().astype('float')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return [image, target]

transform_pipe = transforms.Compose([
    #torchvision.transforms.ToPILImage(), # Convert np array to PILImage
    transforms.Grayscale(num_output_channels=1),
    # Resize image to 224 x 224 as required by most vision models
    transforms.Resize(
        size=(224, 224)
    ),
    
    # Convert PIL image to tensor with image values in [0, 1]
    transforms.ToTensor(),
    
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])        

tranform_target_pipe = transforms.Compose([
    # Convert PIL image to tensor with image values in [0, 1]
])        

train_ds = CelebDataset(csv_file='celeba_10_labels_train.csv',
                                    root_dir='data_cant_touch_this',
                                    transform=transform_pipe,
                                    target_transform=None
                                    )
valid_ds = CelebDataset(csv_file='celeba_10_labels_val.csv',
                                    root_dir='data_cant_touch_this',
                                    transform=transform_pipe,
                                    target_transform=None 
                                    )
test_ds = CelebDataset(csv_file='celeba_10_labels_test.csv',
                                    root_dir='data_cant_touch_this',
                                    transform=transform_pipe,
                                    target_transform=None 
                                    )

batch_size=1000

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=10)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=10)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=10)

print("Done making data loaders!")

#Define a new neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(50176, 10) #Add 1 fully connected layer with 1 neuron

    def forward(self, x):
        x = x.view(x.shape[0],50176)
        x = self.fc(x)
        return x

#display the model summary, check the number of parameters
model = Net()

#Use GPU - IT IS NOT REQUIRED: model = model.cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device) 

num_params=0
for p in model.parameters():
    num_params+=p.numel()
print("The number of parameters {}".format(num_params))

#Actually getting the param/layer weights
print(model.state_dict().keys())
print(model.state_dict()['fc.weight'].shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Optimization params
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)

#Loss
Loss = nn.BCEWithLogitsLoss()

#Start training process
epochs=0

import tqdm
import numpy as np
model.train()
for epoch in range(epochs):
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(train_dl), total=len(train_dl), desc="Epoch: {}".format(epoch)):
        data = data.view(data.shape[0], 50176)
        data = data.to(device) 
        target = target.float().to(device)
        output = model(data)
        print(output.shape)
        optimizer.zero_grad()
        loss = Loss(output,target)
        loss.backward()
        optimizer.step()
        loss_cum.append(loss.item())
        Acc += torch.round(output.data.cpu()).squeeze(1).long().eq(target.data.cpu().long()).sum()
    print("")
    print("Loss: %0.3f"%(np.array(loss_cum).mean()))
    print("Acc: %0.2f"%(float(Acc*100)/len(train_dl.dataset)))

def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        return model

model = load_checkpoint('checkpoint.pth')

predictions = np.zeros((1,10))

for batch_idx, (data,target) in tqdm.tqdm(enumerate(test_dl), total=len(test_dl)):
        data = data.view(data.shape[0], 50176)
        data = data.to(device) 
        target = target.float().to(device)
        output = model(data)
        output = torch.sigmoid(output) >= 0.5
        predictions_batch = output.float().cpu().numpy()
        predictions = np.concatenate((predictions, predictions_batch))
        np.savetxt('predictions.csv', predictions) 

print("Done making predictions!")
np.savetxt("predictions_pretty.txt", predictions, fmt='%5.4g')
