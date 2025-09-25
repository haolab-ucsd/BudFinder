
'''
training script for MAE 
takes in raw pictures for a unaltered mother cell image and reconstruct it
uses the encoder for age prediction later
'''


import torch
import numpy as np
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.tensorboard import SummaryWriter #new tensorboard logger method
import random
import math
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from sklearn.metrics import confusion_matrix
from Models import MAE_2d,TransformerSequence,TransformerBlock,SelfAttentionBlock
from Utils import *
from Datasets import CellCropDataset
import yaml
import argparse

################## MAIN SCRIPT ############################################################

## get yaml config file directory
parse = argparse.ArgumentParser()
parse.add_argument('--config_dir', type=str,
                                    required=True,
                                    default='training_mae_config.yaml',
                                    help='directory of training config yaml file')
args = parse.parse_args()

# open yaml config file
with open(args.config_dir,"r") as file:
    config = yaml.safe_load(file)
    
######### IMPORT DATA ########################
## inputs 
tifpath = config['data']['tifpath'] # path to the raw movie files
csvpath = config['data']['csvpath'] # csv file directory

## training settings
num_epoch = int(config['training']['num_epoch'])
lr =  float(config['training']['lr'])
wdecay = float(config['training']['wdecay'])
warm_steps = int(config['training']['warm_steps'])
cos_total_steps = int(config['training']['cos_total_steps'])
scheduler = config['training']['scheduler'] # None , 'warmup', 'cosine', 'mixed'
decay_val = float(config['training']['decay_val']) # ema decay
T_mult = int(config['training']['T_mult'])
save_freq = int(config['training']['save_freq'])
batch_num = int(config['training']['batch_num']) # batch size
accumulation_steps = int(config['training']['accumulation_steps']) # accumulation steps
pretrained_weight = config['training']['pretrained_weight'] #

## generate training crops
flist = [fname for fname in os.listdir(tifpath) if 'c1' in fname]
print('generating crops....')
df = pd.concat(Parallel(n_jobs=4)(delayed(savemovie_crop)(fname,tifpath,csvpath) for fname in flist))


## create weight folder
weights_path = 'MAEWeights_test/'
os.makedirs(weights_path,exist_ok=True)

# read data
df_og = pd.read_csv(csvpath)

## attach division and idx information into dataframe

# add idx glob information
df['idx_glob'] = df.apply(getidxinfo,axis=1,df_og_=df_og)
df = df[df['idx_glob'].apply(lambda x: len(x)>0)]
df['idx_glob'] = df['idx_glob'].apply(lambda x: x[0])


# split each species dataframe individually
df_train, df_valid = splitTrainValid(df)

# create dataset object
trainset = CellCropDataset(df_train)
validset = CellCropDataset(df_valid)

# create dataloaders
batch_num = 5
trainloader = DataLoader(trainset, batch_size=batch_num, shuffle=True)
validloader = DataLoader(validset, batch_size=batch_num*10, shuffle=True)

# create device variable to load model onto gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(comment='MAE')

##### MODEL INSTANTIATION #################

mae = MAE_2d(in_chan=int(config['model']['in_chan']), # number of input channels
              in_dim=int(config['model']['in_dim']), # input size
              patch_size=int(config['model']['patch_size']), # patch size
              n_enc_layers=int(config['model']['n_enc_layers']), # encoder depth
              n_dec_layers=int(config['model']['n_dec_layers']), # decoder depth
              mask_frac= float(config['model']['mask_frac']), # fraction of mask patches
              n_heads=int(config['model']['n_heads'])).to(device)
mae_EMA = MAE_2d(in_chan=int(config['model']['in_chan']), # number of input channels
              in_dim=int(config['model']['in_dim']), # input size
              patch_size=int(config['model']['patch_size']), # patch size
              n_enc_layers=int(config['model']['n_enc_layers']), # encoder depth
              n_dec_layers=int(config['model']['n_dec_layers']), # decoder depth
              mask_frac= float(config['model']['mask_frac']), # fraction of mask patches
              n_heads=int(config['model']['n_heads'])).to(device)

#### SET UP EMA ################
if pretrained_weight!='None':
    mae.load_state_dict(torch.load(pretrained_weight))
# copy weights from original model to EMA model
mae_EMA.load_state_dict(mae.state_dict())

# set up optimizer
optimizer = torch.optim.AdamW(mae.parameters(),lr=lr,
                                         weight_decay=wdecay)
# create warmup schedulers
warmup_sch = Warmup_sch(optimizer,warm_steps,lr)

# create cosine annealing schedulers
cos_sch = CosineAnnealingWarmRestarts(optimizer,
                                      T_0=cos_total_steps,
                                      T_mult=2)

##### START TRAINING ##########################
count = 0 # keep track of actual number of total iteration
acc_count = 0 # keep track of number of model update iteration
# create iter objects
data_iter_train = iter(trainloader)
data_iter_valid = iter(validloader)

# get batch number
batch_number_train = len(trainloader)
batch_number_valid = len(validloader)


# set the number of accumulation steps
accumulation_steps = 14
acc_step_num = batch_number_train//accumulation_steps

# start training routine

for i in range(num_epoch):
    
    running_loss = 0.0
    
    # train loop    
    for step in range(acc_step_num):
        
        # zero optimizer
        optimizer.zero_grad()
        
        # zero running loss
        running_loss = 0
        
        # set model to train
        mae.train()
        
        for substep in range(accumulation_steps): 
            # extract data
            im_train = next(data_iter_train)
            im_train = im_train.float().to(device)
            
            # get denoiser loss
            
            pred_train = mae(im_train)
            loss_train = F.mse_loss(pred_train,im_train)

            loss_train = loss_train/accumulation_steps # normalize by acc step

            # accumulate loss gradient
            loss_train.backward()

            # add to running loss
            running_loss += loss_train.item()
            
            # update count variable to for validation data calculation
            count += 1
        
        # update model after each accumulation step
        optimizer.step()

        # update EMA
        EMA_updater(mae,mae_EMA,decay_val)
        
        
        # update learning rate schedulers
        if scheduler == 'mixed':
            combine_sch(acc_count+1,warm_steps,warmup_sch, cos_sch)
        if scheduler == 'warmup':
            warmup_sch.step()
        if scheduler == 'cosine':
            cos_sch.step()
        
        # update accumulater count
        acc_count += 1
        
        
        # get validation loss ######################################################
        ############################################################################
        
        with torch.no_grad():
            if (acc_count+1) % batch_number_valid == 0:
                data_iter_valid = iter(validloader)

            # extract data
            im_valid = next(data_iter_valid)

            im_valid = im_valid.float().to(device)
            
            ## get validation loss
            mae_EMA.eval()

            pred_valid = mae_EMA(im_valid)
            loss_valid = F.mse_loss(pred_valid,im_valid)

    
        # print result and save statistics
        if (acc_count+1)%100 == 0:
            # print running loss training
            print('epoch: {} | step {} | running_loss: {}'.format(i,acc_count+1,running_loss))
            print('################')
            print('loss_valid: {} | step {}'.format(loss_valid.item(),acc_count+1,))
            print('################')
            
            writer.add_scalar("train_loss", running_loss,acc_count+1)
            writer.add_scalar("valid_loss", loss_valid.item(),acc_count+1)
            writer.add_image('OG', to_np(im_valid[:10,:,:,:]), count,dataformats='NCHW')
            writer.add_image('reconstructed', to_np(pred_valid[:10,:,:,:]), count,dataformats='NCHW')
    
    #### reset training data iter
    data_iter_train = iter(trainloader)
    # save weights every 10 epch
    if (i+1) % save_freq == 0: 
        torch.save(mae_EMA.state_dict(),weights_path + 'MAE_epoch{}.pt'.format(i+1))
        # save out the encoder-related parts and the position embedding
        torch.save(mae_EMA.encoder,weights_path + 'MAE_encoder_epoch{}.pth'.format(i+1))
        torch.save(mae_EMA.init_enc,weights_path + 'MAE_init_enc_epoch{}.pth'.format(i+1))
        torch.save(mae_EMA.pos_emb,weights_path + 'MAE_pos_emb_epoch{}.pth'.format(i+1))

