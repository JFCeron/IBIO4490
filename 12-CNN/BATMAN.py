"""
Created on Thr Apr 18 08:15:43 2019

@author: BATMAN
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from skimage import io, transform
from PIL import Image

import numpy as np
import pandas as pd

class CelebDataset(Dataset):
    """Celeb dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
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

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.open(img_name)
        
        target = self.landmarks_frame.iloc[idx, 1].as_matrix().astype('float')
        #labels = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        #labels = labels.astype('float')
        #sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return [image, target]

transform_pipe = torchvision.transforms.Compose([
    #torchvision.transforms.ToPILImage(), # Convert np array to PILImage
    
    # Resize image to 224 x 224 as required by most vision models
    torchvision.transforms.Resize(
        size=(224, 224)
    ),
    
    # Convert PIL image to tensor with image values in [0, 1]
    torchvision.transforms.ToTensor(),
    
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

celeb_dataset = CelebDataset(csv_file='annotations_Bangs.csv',
                                    root_dir='img_align_celeba',
                                    transform=None 
                                    )

fig = plt.figure()

for i in range(len(celeb_dataset)):
    sample = celebt_dataset[i]

    print(i, sample['image'].shape, sample['labels'])

    ax = plt.subplot(1, 4, i + 1)
    plt.imshow(sample['image'])
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')

    if i == 3:
        plt.show()
        fig.savefig('test_celeb.png')
        break                                    

# Load data into train and test sets
#train = datasets.MNIST('data', train=True, download=True)
#test = datasets.MNIST('data', train=False)

batch_size=128

transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
data_train = celeb_dataset 
#datasets('data', train=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)

#data_test = datasets.MNIST('data', train=True, transform=transform_train)
#test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True)

# Thank you Alexis!!!
alexnet = models.alexnet(pretrained=True)
print(alexnet)

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

#Optimization params
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)

#Loss
Loss = nn.MSELoss()

#Start training process
epochs=20
number_of_pixels = 218*178*3

import tqdm

model.train()
for epoch in range(epochs):
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc="Epoch: {}".format(epoch)):
        data = data.view(-1,number_of_pixels)
        data = data.to(device) 
        target = target.float().to(device)

        output = model(data)
        optimizer.zero_grad()
        #loss = Loss(output,target)
        loss = F.mse_loss(output, target) #Virtually the same as nn.MSELoss
        loss.backward()
        optimizer.step()
        loss_cum.append(loss.item())
        Acc += torch.round(output.data.cpu()).squeeze(1).long().eq(target.data.cpu().long()).sum()
    print("")
    print("Loss: %0.3f"%(np.array(loss_cum).mean()))
    print("Acc: %0.2f"%(float(Acc*100)/len(train_loader.dataset)))