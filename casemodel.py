#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%fix model
import torch
from torchsummary import summary
import torch.nn as nn
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') #GPU 할당
print(device)
#%%
class casemodel(nn.Module):
    def __init__(self):
        super(casemodel, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )
        self.fc2=nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )
        self.flatten=nn.Sequential(
            nn.Flatten()
            )
        self.fc3=nn.Sequential(
            nn.Linear(13824,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,4)
            )
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.flatten(x)
        x = self.fc3(x)
        return x
#%%
"""
model=casemodel()
model=model.to(device)
model.eval()
summary(model, input_size=(1,72,48), device=device.type)
"""