#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
#%%train class
def train(model, modeltype, optimizer, train_loader, val_loader, scheduler, device, epoch=0,tr_loss=[],vali_loss=[],best_rmse=999999,last_epoch=0):
    best_rmse=9999
    model.cuda() #set cuda
    criterion = nn.L1Loss().to(device) #loss function as cuda
    a=last_epoch
    tr_loss=[]
    vali_loss=[]
    for epoch in range(1, epoch):
        model.train()
        train_loss = []
        for sem, depth in tqdm(iter(train_loader)):
            
                
            sem = sem.float().to(device)
            depth = depth.float().to(device)
            
            optimizer.zero_grad()
            model_pred = model(sem)
            
            loss = criterion(model_pred, depth)
            
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        
        val_loss,val_rmse = validation(model, nn.MSELoss().to(device), val_loader, device)
        print(f'\nEpoch : [{epoch+a}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}] Val rmse : [{val_rmse:.5f}]')
        print(f'model tpye : [{modeltype}]')
        tr_loss.append(np.mean(train_loss))
        vali_loss.append(val_loss)
        if best_rmse > val_rmse:
            best_rmse = val_rmse
            if modeltype==1:
                torch.save(model.state_dict(),"/home/nvidia/Workspace/Samsung/best_model1.pth")
            if modeltype==2:
                torch.save(model.state_dict(),"/home/nvidia/Workspace/Samsung/best_model2.pth")
            if modeltype==3:
                torch.save(model.state_dict(),"/home/nvidia/Workspace/Samsung/best_model3.pth")
            if modeltype==4:
                torch.save(model.state_dict(),"/home/nvidia/Workspace/Samsung/best_model4.pth")
            
            print('Model Saved\n')
        
        
        if scheduler is not None:
            scheduler.step()
    return tr_loss,vali_loss,val_rmse,epoch
#%%validation class
def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    val_rmse=[]
    with torch.no_grad():
        for sem, depth in tqdm(iter(val_loader)):
            sem = sem.float().to(device)
            depth = depth.float().to(device)
            model_pred = model(sem)
            loss = criterion(model_pred, depth)
            val_loss.append(loss.item())
            
            pred=(model_pred*255.).type(torch.int8).float()
            true=(depth*255.).type(torch.int8).float()
            rmse_loss=torch.sqrt(criterion(pred, true))
            val_rmse.append(rmse_loss.item())
                
            
    return np.mean(val_loss),np.mean(val_rmse)

#%%
def casetrain(model, optimizer, criterion, train_loader,val_loader,epochs, scheduler, device, bestacc=0.0): 
    
    model.cuda() #set cuda    
    best_acc=bestacc
    best_apoch=0 
    tr_loss = []
    val_loss=[]
    #Loss Function ??????
    #criterion = nn.CrossEntropyLoss().to(device)
 
    for epoch in range(1,epochs+1): #????????? ??????
        model.train() #?????? ??????
        running_loss = 0.0
            
        for img, label in tqdm(iter(train_loader)):
            img, label = img.to(device), label.to(device) #?????? ?????????
            optimizer.zero_grad() #???????????? optimizer ?????????
        
            # Data -> Model -> Output
            logit = model(img) #????????? ??????
            loss = criterion(logit, label) #???????????? ??????
            
            # ?????????
            loss.backward() #???????????? ?????? ????????? 
            optimizer.step() #????????? ?????????
            running_loss += loss.item()
            train_loss=running_loss / len(train_loader)  
        print('[%d] Train loss: %.10f' %(epoch, running_loss / len(train_loader)))
        tr_loss.append(running_loss / len(train_loader))
        if scheduler is not None:
            scheduler.step()
            
        #Validation set ??????
        model.eval() #evaluation ???????????? ???????????? ????????? ?????? layer?????? ????????? off ???????????? ?????? ??????
        vali_loss = 0.0
        correct = 0
        
        with torch.no_grad(): #???????????? ???????????? ????????? ????????? no_grad ??????
            for img, label in tqdm(iter(val_loader)):
                img, label = img.to(device), label.to(device)

                logit = model(img)
                vali_loss += criterion(logit, label)
                pred = logit.argmax(dim=1, keepdim=True)  
                correct += pred.eq(label.view_as(pred)).sum().item() #???????????? ???????????? ????????? 1 ????????? 0?????? ??????
        vali_acc = 100.0 * correct / len(val_loader.dataset)*1.0
        print('Vail set: Loss: {:.4f}, Accuracy: {}/{} ( {:.1f}%)\n'.format(vali_loss / len(val_loader), correct, len(val_loader.dataset), 100.0 * correct / len(val_loader.dataset)*1.0))
        val_loss.append(vali_loss / len(val_loader))
        #????????? ?????? ??????
        if best_acc < vali_acc:
            best_acc = vali_acc
            torch.save(model.state_dict(), '/home/nvidia/Workspace/Samsung/case_best_model.pth') #??? ??????????????? best_model.pth??? ??????
            print('Model Saved.')
    return best_acc,best_apoch,vali_acc,val_loss,tr_loss