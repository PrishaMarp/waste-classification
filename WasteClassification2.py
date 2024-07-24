#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.autograd import Variable
class ConvNet(nn.Module):
    def __init__(self, num_classes = 2):
        super(ConvNet, self).__init__()

        #(64,3,150,150) 
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1) #(64,12,150,150)
        # self.bn1 = nn.BatchNorm2d(num_features=12) #(64,12,150,150)
        # self.relu1 = nn.ReLU() #(64,12,150,150)
        # self.pool = nn.MaxPool2d(kernel_size=2) #(64,12,75,75)

        # self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1) #(64,20,75,75)
        # self.relu2 = nn.ReLU() #(64,20,75,75)

        # self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1) #(64,32,75,75)
        # self.bn3 = nn.BatchNorm2d(num_features=32) #(64,32,75,75)
        # self.relu3 = nn.ReLU() #(64,32,75,75)

        # self.fc = nn.Linear(in_features=32*75*75, out_features=2)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1) #(64,12,150,150)
        self.bn1 = nn.BatchNorm2d(num_features=12) #(64,12,150,150)
        self.relu1 = nn.ReLU() #(64,12,150,150)
        self.pool = nn.MaxPool2d(kernel_size=2) #(64,12,75,75)

        self.conv3 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, stride=1, padding=1) #(64,32,75,75)
        self.bn3 = nn.BatchNorm2d(num_features=32) #(64,32,75,75)
        self.relu3 = nn.ReLU() #(64,32,75,75)

        self.fc = nn.Linear(in_features=32*16*16, out_features=2)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)

        #output = self.conv2(output)
        #output = self.relu2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = output.view(-1,32*16*16)
        output = self.fc(output)
        return output


# In[2]:


model1 = ConvNet(num_classes=2)
model1 = model1.to('cpu')
print(model1)

