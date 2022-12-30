#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import torch
from tqdm import tqdm
import numpy as np
#%%inference classF
import zipfile
def inference(model1,model2,model3,model4,case_model,case_loader, test_loader, device):
    model1.to(device)
    model1.eval()
    model2.to(device)
    model2.eval()
    model3.to(device)
    model3.eval()
    model4.to(device)
    model4.eval()
    case_model.to(device)
    case_model.eval()
    a1=torch.tensor([[0]]).to(device)
    a2=torch.tensor([[1]]).to(device)
    a3=torch.tensor([[2]]).to(device)
    a4=torch.tensor([[3]]).to(device)
    result_name_list = []
    result_list = []
    a=0
    b=0
    c=0
    d=0
    with torch.no_grad():
        for sem, name in tqdm(iter(test_loader)):
            
            sem_img = sem.float().to(device)
            case_sem=sem.to(device)

            logit = case_model(case_sem)
            case = logit.argmax(dim=1, keepdim=True)
            global model_pred
            if case==a1:
                model_pred = model1(sem_img)
                a+=1
                case_int=1
            elif case==a2:
                model_pred = model2(sem_img)
                b+=1
                case_int=2
            elif case==a3:
                model_pred = model3(sem_img)
                c+=1
                case_int=3
            elif case==a4:
                model_pred = model4(sem_img)
                d+=1
                case_int=4
            else:
                print('something wrong')
            for pred, img_name in zip(model_pred, name):
                pred = pred.cpu().numpy().transpose(1,2,0)*255.0
                pred=pred.astype(int)
                Min_value1=np.min(pred[0])
                Min_value2=np.min(pred[71])
                Min_value=min(Min_value1,Min_value2)
                Class=case_int*10+130
                pred[pred>=Min_value]=Class #define background brighter than min value from first line
                pred[pred>Class]=Class #define background brighter than class value
                save_img_path =f'{img_name}'
                #im=torchvision.transforms.functional.to_pil_image(pred)
                #im.save(save_img_path)
                cv2.imwrite(save_img_path, pred)
                result_name_list.append(save_img_path)
                result_list.append(pred)
    os.makedirs('/home/nvidia/Workspace/Samsung/submission', exist_ok=True)
    os.chdir("/home/nvidia/Workspace/Samsung/submission/")
    sub_imgs = []
    for path, pred_img in zip(result_name_list, result_list):
        cv2.imwrite(path, pred_img)
        sub_imgs.append(path)
    submission = zipfile.ZipFile("/home/nvidia/Workspace/Samsung/submission/the_last.zip", 'w')
    case_list=[a,b,c,d]
    print(case_list)
    for path in sub_imgs:
        submission.write(path)
    submission.close()