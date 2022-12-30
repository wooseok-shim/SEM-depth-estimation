#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:44:59 2022

@author: nvidia
"""
#%%dataset class
import random
import numpy as np
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
#%%
original_transtorm=transforms.ToTensor()
horizontal_transform=transforms.Compose([
    transforms.RandomHorizontalFlip(1.0),
    transforms.ToTensor()
    ])

vertical_transform=transforms.Compose([
    transforms.RandomVerticalFlip(1.0),
    transforms.ToTensor()
    ])

both_transform=transforms.Compose([
    transforms.RandomHorizontalFlip(1.0),
    transforms.RandomVerticalFlip(1.0),
    transforms.ToTensor()
    ])
#%%
class CustomDataset(Dataset):
    def __init__(self, sem_path_list, depth_path_list,transform=None,train=True):
        self.sem_path_list = sem_path_list
        self.depth_path_list = depth_path_list
        self.transform=transform
        self.train=train
    def __getitem__(self, index):
        sem_path = self.sem_path_list[index]
        sem_img=cv2.imread(sem_path)
        if np.shape(np.array(sem_img))==(72,48,3):
            sem_img=sem_img[:,:,0]
        if self.train==True:
            temp = cv2.GaussianBlur(sem_img, (0, 0), 1)
            for i in range(72):
                for j in range(48):
                    if temp[i,j] >50:
                        temp[i,j]=temp[i,j]*1.15
                    if temp[i,j] <50:
                        temp[i,j]=temp[i,j]+4
            sem_img=np.array(temp)
            sem_img=Image.fromarray(sem_img)
        else:
            sem_img=np.array(sem_img)
            sem_img=Image.fromarray(sem_img)
            
        if self.depth_path_list is not None:
            depth_path = self.depth_path_list[index]
            depth_img=Image.open(depth_path)
            if self.transform is not None:
                sem_img=self.transform(sem_img)
                depth_img=self.transform(depth_img)
            return sem_img, depth_img # B,C,H,W
        else:
            img_name = sem_path.split('/')[-1]
            if self.transform is not None:
                sem_img=self.transform(sem_img)
            return sem_img, img_name # C,B,H,W.unsqueeze(0)
        
    def __len__(self):
        return len(self.sem_path_list)
#%%
simulation_sem_paths1 = sorted(glob.glob('/home/nvidia/Workspace/Samsung/simulation_data/SEM/Case_1/*/*.png'))
simulation_depth_paths1 = sorted(glob.glob('/home/nvidia/Workspace/Samsung/simulation_data/Depth/Case_1/*/*.png')+glob.glob('/home/nvidia/Workspace/Samsung/simulation_data/Depth/*/*/*.png'))
simulation_sem_paths2 = sorted(glob.glob('/home/nvidia/Workspace/Samsung/simulation_data/SEM/Case_2/*/*.png'))
simulation_depth_paths2 = sorted(glob.glob('/home/nvidia/Workspace/Samsung/simulation_data/Depth/Case_2/*/*.png')+glob.glob('/home/nvidia/Workspace/Samsung/simulation_data/Depth/*/*/*.png'))
simulation_sem_paths3 = sorted(glob.glob('/home/nvidia/Workspace/Samsung/simulation_data/SEM/Case_3/*/*.png'))
simulation_depth_paths3 = sorted(glob.glob('/home/nvidia/Workspace/Samsung/simulation_data/Depth/Case_3/*/*.png')+glob.glob('/home/nvidia/Workspace/Samsung/simulation_data/Depth/*/*/*.png'))
simulation_sem_paths4 = sorted(glob.glob('/home/nvidia/Workspace/Samsung/simulation_data/SEM/Case_4/*/*.png'))
simulation_depth_paths4 = sorted(glob.glob('/home/nvidia/Workspace/Samsung/simulation_data/Depth/Case_4/*/*.png')+glob.glob('/home/nvidia/Workspace/Samsung/simulation_data/Depth/*/*/*.png'))


test_sem_path_list = sorted(glob.glob('/home/nvidia/Workspace/Samsung/test/SEM/*.png'))
#%%checking datalen
data_len1 = len(simulation_sem_paths1)
simul_len1=len(simulation_depth_paths1)
data_len2 = len(simulation_sem_paths2)
simul_len2=len(simulation_depth_paths2)
data_len3 = len(simulation_sem_paths3)
simul_len3=len(simulation_depth_paths3)
data_len4 = len(simulation_sem_paths4)
simul_len4=len(simulation_depth_paths4)

#%%split
random.Random(19991006).shuffle(simulation_sem_paths1)
random.Random(19991006).shuffle(simulation_depth_paths1)
random.Random(19991006).shuffle(simulation_sem_paths2)
random.Random(19991006).shuffle(simulation_depth_paths2)
random.Random(19991006).shuffle(simulation_sem_paths3)
random.Random(19991006).shuffle(simulation_depth_paths3)
random.Random(19991006).shuffle(simulation_sem_paths4)
random.Random(19991006).shuffle(simulation_depth_paths4)

train_sem_paths1 = simulation_sem_paths1[:int(data_len1*0.9)]
train_depth_paths1 = simulation_depth_paths1[:int(data_len1*0.9)]

val_sem_paths1 = simulation_sem_paths1[int(data_len1*0.9):]
val_depth_paths1 = simulation_depth_paths1[int(data_len1*0.9):]

train_sem_paths2 = simulation_sem_paths2[:int(data_len2*0.9)]
train_depth_paths2 = simulation_depth_paths2[:int(data_len2*0.9)]

val_sem_paths2 = simulation_sem_paths2[int(data_len2*0.9):]
val_depth_paths2 = simulation_depth_paths2[int(data_len2*0.9):]

train_sem_paths3 = simulation_sem_paths3[:int(data_len3*0.9)]
train_depth_paths3 = simulation_depth_paths3[:int(data_len3*0.9)]

val_sem_paths3 = simulation_sem_paths3[int(data_len3*0.9):]
val_depth_paths3 = simulation_depth_paths3[int(data_len3*0.9):]

train_sem_paths4 = simulation_sem_paths4[:int(data_len4*0.9)]
train_depth_paths4 = simulation_depth_paths4[:int(data_len4*0.9)]

val_sem_paths4 = simulation_sem_paths4[int(data_len4*0.9):]
val_depth_paths4 = simulation_depth_paths4[int(data_len4*0.9):]

#%%make dataset
tr11 = CustomDataset(train_sem_paths1, train_depth_paths1,original_transtorm)
tr12 = CustomDataset(train_sem_paths1, train_depth_paths1,horizontal_transform)
tr13 = CustomDataset(train_sem_paths1, train_depth_paths1,vertical_transform)
tr14 = CustomDataset(train_sem_paths1, train_depth_paths1,both_transform)
train_dataset1=tr11+tr12+tr13+tr14
train_loader1 = DataLoader(train_dataset1, batch_size = 64, shuffle=True, num_workers=16)



val11 = CustomDataset(val_sem_paths1, val_depth_paths1,original_transtorm)
val12 = CustomDataset(val_sem_paths1, val_depth_paths1,horizontal_transform)
val13 = CustomDataset(val_sem_paths1, val_depth_paths1,vertical_transform)
val14 = CustomDataset(val_sem_paths1, val_depth_paths1,both_transform)
val_dataset1 =val11+val12+val13+val14
val_loader1 = DataLoader(val_dataset1, batch_size=64, shuffle=False, num_workers=16)



tr21 = CustomDataset(train_sem_paths2, train_depth_paths2,original_transtorm)
tr22 = CustomDataset(train_sem_paths2, train_depth_paths2,horizontal_transform)
tr23 = CustomDataset(train_sem_paths2, train_depth_paths2,vertical_transform)
tr24= CustomDataset(train_sem_paths2, train_depth_paths2,both_transform)
train_dataset2=tr21+tr22+tr23+tr24
train_loader2 = DataLoader(train_dataset2, batch_size = 64, shuffle=True, num_workers=16)



val21 = CustomDataset(val_sem_paths2, val_depth_paths2,original_transtorm)
val22 = CustomDataset(val_sem_paths2, val_depth_paths2,horizontal_transform)
val23 = CustomDataset(val_sem_paths2, val_depth_paths2,vertical_transform)
val24 = CustomDataset(val_sem_paths2, val_depth_paths2,both_transform)
val_dataset2 =val21+val22+val23+val24
val_loader2 = DataLoader(val_dataset2, batch_size=64, shuffle=False, num_workers=16)



tr31 = CustomDataset(train_sem_paths3, train_depth_paths3,original_transtorm)
tr32 = CustomDataset(train_sem_paths3, train_depth_paths3,vertical_transform)
tr33 = CustomDataset(train_sem_paths3, train_depth_paths3,horizontal_transform)
tr34 = CustomDataset(train_sem_paths3, train_depth_paths3,both_transform)
train_dataset3=tr31+tr32+tr33+tr34
train_loader3 = DataLoader(train_dataset3, batch_size = 64, shuffle=True, num_workers=16)



val31 = CustomDataset(val_sem_paths3, val_depth_paths3,original_transtorm)
val32 = CustomDataset(val_sem_paths3, val_depth_paths3,vertical_transform)
val33 = CustomDataset(val_sem_paths3, val_depth_paths3,horizontal_transform)
val34 = CustomDataset(val_sem_paths3, val_depth_paths3,both_transform)
val_dataset3 =val31+val32+val33+val34
val_loader3 = DataLoader(val_dataset3, batch_size=64, shuffle=False, num_workers=16)



tr41 = CustomDataset(train_sem_paths4, train_depth_paths4,original_transtorm)
tr42 = CustomDataset(train_sem_paths4, train_depth_paths4,vertical_transform)
tr43 = CustomDataset(train_sem_paths4, train_depth_paths4,horizontal_transform)
tr44 = CustomDataset(train_sem_paths4, train_depth_paths4,both_transform)
train_dataset4=tr41+tr42+tr43+tr44
train_loader4 = DataLoader(train_dataset4, batch_size = 64, shuffle=True, num_workers=16)



val41 = CustomDataset(val_sem_paths4, val_depth_paths4,original_transtorm)
val42 = CustomDataset(val_sem_paths4, val_depth_paths4,vertical_transform)
val43 = CustomDataset(val_sem_paths4, val_depth_paths4,horizontal_transform)
val44 = CustomDataset(val_sem_paths4, val_depth_paths4,both_transform)
val_dataset4 =val41+val42+val43+val44
val_loader4 = DataLoader(val_dataset4, batch_size=64, shuffle=False, num_workers=16)


#%%
class CaseDataset(Dataset):
    def __init__(self, sem_path_list,transform = None):
        self.transform=transform
        self.sem_path_list = sem_path_list
    def __getitem__(self, index):
        sem_path = self.sem_path_list[index]
        sem_img=cv2.imread(sem_path)
        if np.shape(np.array(sem_img))==(72,48,3):
            sem_img=sem_img[:,:,0]
        
        sem_img=np.array(sem_img)
        sem_img=Image.fromarray(sem_img)
        if sem_path.split('/')[-3]=="Depth_110":
            case_name = 0
            
        elif sem_path.split('/')[-3]=="Depth_120":
            case_name = 1
        
        elif sem_path.split('/')[-3]=="Depth_130":
            case_name = 2
        
        elif sem_path.split('/')[-3]=="Depth_140":
            case_name = 3
        else:
            case_name = 10
        if self.transform is not None:
            sem_img=self.transform(sem_img)
        return sem_img, case_name # C,B,H,W.unsqueeze(0)       
    def __len__(self):
        return len(self.sem_path_list)
#%%
caselist=sorted(glob.glob("/home/nvidia/Workspace/Samsung/train/SEM/*/*/*.png"))
case_data_len = len(caselist)
random.Random(19991006).shuffle(caselist)
train_case_paths=caselist[:54598]
validation_case_paths=caselist[54598:]

trcase1=CaseDataset(train_case_paths,original_transtorm)
trcase2=CaseDataset(train_case_paths,horizontal_transform)
trcase3=CaseDataset(train_case_paths,vertical_transform)
trcase4=CaseDataset(train_case_paths,both_transform)

case_train_dataset=trcase1+trcase2+trcase3+trcase4


valcase1=CaseDataset(validation_case_paths,original_transtorm)
valcase2=CaseDataset(validation_case_paths,horizontal_transform)
valcase3=CaseDataset(validation_case_paths,vertical_transform)
valcase4=CaseDataset(validation_case_paths,both_transform)

case_validation_dataset=valcase1+valcase2+valcase3+valcase4


case_train_loader = DataLoader(case_train_dataset, batch_size = 64, shuffle=True, num_workers=16)
case_validation_loader=DataLoader(case_validation_dataset, batch_size = 64, shuffle=True, num_workers=16)
#%%
testlist=sorted(glob.glob("/home/nvidia/Workspace/Samsung/test/SEM/*.png"))
case_test_dataset=CaseDataset(testlist,original_transtorm)
case_test_loader=DataLoader(case_test_dataset,batch_size=1,shuffle=False,num_workers=16)

test_dataset = CustomDataset(test_sem_path_list, None,original_transtorm,False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)

    
    
    
    
    
    