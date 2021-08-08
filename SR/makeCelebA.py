import csv
import torch
import os
from datasets import HelenDataset, CelebDataset, FERDataset
from utils import write_log, _normalize, _denormalize
from PIL import Image

trn_dataset = CelebDataset(mode='train')
val_dataset = CelebDataset(mode='test')
trn_dloader = torch.utils.data.DataLoader(dataset=trn_dataset, batch_size=1, shuffle=False)
val_dloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

LR_output_dir = '../CelebA/LR/'
HR_output_dir='../CelebA/HR/'
for batch_idx, (image_lr, image_hr,_, _) in enumerate(trn_dloader, start=1):
    #print(image_hr.shape)
    image_lr = torch.from_numpy(_normalize(image_lr)).float().cuda()
    image_hr=torch.from_numpy(_normalize(image_hr)).float().cuda()




    real_lr = _denormalize(image_lr)[0].astype('uint8')
    real_hr=_denormalize(image_hr)[0].astype('uint8')


    real_lrimg = Image.fromarray(real_lr)
    real_hrimg=Image.fromarray(real_hr)

    real_lrimg.save(os.path.join(LR_output_dir, '%d.jpg' % batch_idx))
    real_hrimg.save(os.path.join(HR_output_dir, '%d.jpg'% batch_idx))

