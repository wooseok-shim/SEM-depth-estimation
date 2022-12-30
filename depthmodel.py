#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:46:57 2022

@author: nvidia
"""
#%%model class
import torch.nn as nn
class depthmodel(nn.Module):
    def __init__(self):
        super(depthmodel,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,64,3, stride=2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,128,3, stride=2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,256,3,stride=2, padding = 1),
            nn.ReLU()
            )
        
        self.flatten = nn.Sequential(nn.Flatten(start_dim = 1),
                                     nn.Dropout(0.2))

        self.encoder_lin = nn.Sequential(
            nn.Linear(256*9*6,2048),
            nn.ReLU(),
            nn.Linear(2048, 256*9*6),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size = (256,9,6))
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256,128,3,stride=2,padding = 1,output_padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,3, stride=2, padding = 1, output_padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,1,3, stride=2, padding = 1, output_padding = 1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder(x)                                                     
        #CNN->flatten->fully connected layer->unflatten->upsampling cnn
        return x