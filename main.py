#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%import
import torch
import warnings
warnings.filterwarnings(action='ignore') 
#%%
import sys
sys.path.append("/home/nvidia/Workspace/Samsung")
from seed import seed_everything
from customdataset import *
from depthmodel import depthmodel
from train import train,casetrain
from inference import inference
from casemodel import casemodel
#%%device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
#%%CFG
CFG = {
    'WIDTH':48,
    'HEIGHT':72,
    'EPOCHS':40,
    'LEARNING_RATE':1e-3,
    'CASE_LEARNING_RATE':1e-4,
    'CASE_EPOCHS':20,
    'BATCH_SIZE':64,
    'SEED':44
}
#%%Random seed
seed_everything(CFG['SEED']) # Seed 고정

#%%model setting
case_model=casemodel()
case_model=torch.nn.DataParallel(case_model)
case_model=case_model.to(device)
case_model.eval()

model1 = depthmodel()
model1=torch.nn.DataParallel(model1)
model1 = model1.to(device)
model1.eval()

model2 = depthmodel()
model2=torch.nn.DataParallel(model2)
model2 = model2.to(device)
model2.eval()

model3 = depthmodel()
model3=torch.nn.DataParallel(model3)
model3 = model3.to(device)
model3.eval()

model4 = depthmodel()
model4=torch.nn.DataParallel(model4)
model4 = model4.to(device)
model4.eval()

#%%parameters setting
optimizer1 = torch.optim.Adam(params = model1.parameters(), lr = CFG["LEARNING_RATE"])
optimizer2 = torch.optim.Adam(params = model2.parameters(), lr = CFG["LEARNING_RATE"])
optimizer3 = torch.optim.Adam(params = model3.parameters(), lr = CFG["LEARNING_RATE"])
optimizer4 = torch.optim.Adam(params = model4.parameters(), lr = CFG["LEARNING_RATE"])

case_optimizer=torch.optim.Adam(params = case_model.parameters(), lr = CFG["CASE_LEARNING_RATE"],weight_decay=0.00001)
case_criterion=torch.nn.CrossEntropyLoss()
scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda=lambda epoch:0.9**epoch)
scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, lr_lambda=lambda epoch:0.9**epoch)
scheduler3 = torch.optim.lr_scheduler.LambdaLR(optimizer3, lr_lambda=lambda epoch:0.9**epoch)
scheduler4 = torch.optim.lr_scheduler.LambdaLR(optimizer4, lr_lambda=lambda epoch:0.9**epoch)
#%%
best_acc,best_epoch,vali_acc,val_loss,tr_loss=casetrain(case_model,case_optimizer,case_criterion,case_train_loader,case_validation_loader,CFG['CASE_EPOCHS'],scheduler1,device)
#best_acc,best_epoch,vali_acc,val_loss,tr_loss=casetrain(case_model,case_optimizer,case_criterion,case_train_loader,case_validation_loader,CFG['EPOCHS'],scheduler,device,best_acc)
#%%visible result
valid_loss=[]
for i in range(len(val_loss)):
    valid_loss.append(val_loss[i].tolist())
    
#%%
import matplotlib.pyplot as plt
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
plt.plot(tr_loss[:],label='train loss')
plt.plot(valid_loss[:],label='validation loss')
plt.legend(loc='best')
plt.show()
#%%
checkpoint=torch.load('/home/nvidia/Workspace/Samsung/best_model1.pth')
model1=depthmodel()
model1=torch.nn.DataParallel(model1)
model1=model1.to(device)
model1.load_state_dict(checkpoint)


checkpoint=torch.load('/home/nvidia/Workspace/Samsung/best_model2.pth')
model2=depthmodel()
model2=torch.nn.DataParallel(model2)
model2=model2.to(device)
model2.load_state_dict(checkpoint)


checkpoint=torch.load('/home/nvidia/Workspace/Samsung/best_model3.pth')
model3=depthmodel()
model3=torch.nn.DataParallel(model3)
model3=model3.to(device)
model3.load_state_dict(checkpoint)


checkpoint=torch.load('/home/nvidia/Workspace/Samsung/best_model4.pth')
model4=depthmodel()
model4=torch.nn.DataParallel(model4)
model4=model4.to(device)
model4.load_state_dict(checkpoint)

#%%
case_checkpoint=torch.load('/home/nvidia/Workspace/Samsung/case_best_model.pth')
case_model=casemodel()
case_model=torch.nn.DataParallel(case_model)
case_model=case_model.to(device)
case_model.load_state_dict(case_checkpoint)
#%%each train
#train first time
train_loss1,validation_loss1,val_rmse1,epochs1 = train(model1,1, optimizer1, train_loader1, val_loader1, scheduler1, device, CFG['EPOCHS'])
train_loss2,validation_loss2,val_rmse2,epochs2 = train(model2,2, optimizer2, train_loader2, val_loader2, scheduler2, device, CFG['EPOCHS'])
train_loss3,validation_loss3,val_rmse3,epochs3 = train(model3,3, optimizer3, train_loader3, val_loader3, scheduler3, device, CFG['EPOCHS'])
train_loss4,validation_loss4,val_rmse4,epochs4 = train(model4,4, optimizer4, train_loader4, val_loader4, scheduler4, device, CFG['EPOCHS'])

#train after second time
"""
train_loss1,validation_loss,best_loss,epochs = train(model1,1, optimizer1, train_loader1, val_loader1, scheduler1, device, CFG['EPOCHS'], train_loss1, validation_loss1, best_loss1, epochs1)
train_loss2,validation_loss,best_loss,epochs = train(model2,2, optimizer2, train_loader2, val_loader2, scheduler2, device, CFG['EPOCHS'], train_loss2, validation_loss2, best_loss2, epochs2)
train_loss3,validation_loss,best_loss,epochs = train(model3,3, optimizer3, train_loader3, val_loader3, scheduler3, device, CFG['EPOCHS'], train_loss3, validation_loss3, best_loss3, epochs3)
train_loss4,validation_loss,best_loss,epochs = train(model4,4, optimizer4, train_loader4, val_loader4, scheduler4, device, CFG['EPOCHS'], train_loss4, validation_loss4, best_loss4, epochs4)
"""

#%%visible result
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()
print(validation_loss1)
plt.plot(train_loss1[:],label='train loss1')
plt.plot(train_loss2[:],label='train loss2')
plt.plot(train_loss3[:],label='train loss3')
plt.plot(train_loss4[:],label='train loss4')
"""
plt.plot(validation_loss1[:],label='validation loss1')
plt.plot(validation_loss2[:],label='validation loss2')
plt.plot(validation_loss3[:],label='validation loss3')
plt.plot(validation_loss4[:],label='validation loss4')
"""
plt.legend(loc='best')
plt.show()
#%%get best models
checkpoint=torch.load('/home/nvidia/Workspace/Samsung/best_model1.pth')
model1=depthmodel()
model1=torch.nn.DataParallel(model1)
model1=model1.to(device)
model1.load_state_dict(checkpoint)


checkpoint=torch.load('/home/nvidia/Workspace/Samsung/best_model2.pth')
model2=depthmodel()
model2=torch.nn.DataParallel(model2)
model2=model2.to(device)
model2.load_state_dict(checkpoint)


checkpoint=torch.load('/home/nvidia/Workspace/Samsung/best_model3.pth')
model3=depthmodel()
model3=torch.nn.DataParallel(model3)
model3=model3.to(device)
model3.load_state_dict(checkpoint)


checkpoint=torch.load('/home/nvidia/Workspace/Samsung/best_model4.pth')
model4=depthmodel()
model4=torch.nn.DataParallel(model4)
model4=model4.to(device)
model4.load_state_dict(checkpoint)


case_checkpoint=torch.load('/home/nvidia/Workspace/Samsung/case_best_model.pth')
case_model=casemodel()
case_model=torch.nn.DataParallel(case_model)
case_model=case_model.to(device)
case_model.load_state_dict(case_checkpoint)
#%%submit!!
inference(model1,model2,model3,model4,case_model,case_test_loader, test_loader, device)
#%%summary

