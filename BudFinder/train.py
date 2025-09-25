
'''
training script for Transformer-based cell division prediction
use the encoder trained from a MAE model scheme
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
from sklearn.metrics import confusion_matrix
from Models import CELLDIV_MAE,TransformerSequence,TransformerBlock,SelfAttentionBlock
from Utils import *
from Datasets import YeastBuddingDataset
import yaml
import argparse


################# MAIN SCRIPT ############################################################

## get yaml config file directory
parse = argparse.ArgumentParser()
parse.add_argument('--config_dir', type=str,
                                    required=True,
                                    default='training_config.yaml',
                                    help='directory of training config yaml file')
args = parse.parse_args()

## open yaml config file
with open(args.config_dir,"r") as file:
    config = yaml.safe_load(file)
    
## inputs extraction
tifpath = config['data']['moviepath']  # path to the raw movie files
csvpath = config['data']['csvpath'] # csv file directory

## masked autoencoder weight extraction
mae_init_enc_path = config['mae_model']['mae_init_enc_path'] # directory of mae initial encoder
mae_pos_emb_path = config['mae_model']['mae_pos_emb_path'] # directory of mae position embedding
mae_enc_path = config['mae_model']['mae_enc_path'] # directory of mae encoder

## training settings
pretrained_weight = config['training']['pretrained_weight'] # directory of pretrained weight, if none then train from scratch
fine_tune = config['training']['fine_tune'] # specify whether you want this to be finetune only
save_freq = int(config['training']['save_freq']) # frequency of epochs when saving occurs
device_placed = config['training']['device_placed'] # device that the exported model will be placed on

num_epoch = int(config['training']['num_epoch'])
lr =  float(config['training']['lr'])
wdecay = float(config['training']['wdecay'])
warm_steps = int(config['training']['warm_steps'])
cos_total_steps = int(config['training']['cos_total_steps'])
scheduler = config['training']['scheduler'] # None , 'warmup', 'cosine', 'mixed'
decay_val = float(config['training']['decay_val']) # ema decay
T_mult = int(config['training']['T_mult'])
batch_num = int(config['training']['batch_num']) # batch size
accumulation_steps = int(config['training']['accumulation_steps']) # accumulation steps

## generate training stacks
flist = [fname for fname in os.listdir(tifpath) if 'c1' in fname]
print('generating stacks....')
df = pd.concat(Parallel(n_jobs=4)(delayed(savemovie)(fname,tifpath,csvpath) for fname in flist))

## create weight folder
weights_path = 'MAEDIVWeights/'
os.makedirs(weights_path,exist_ok=True)

## read og data
df_og = pd.read_csv(csvpath)

## attach division and idx information into dataframe
# add division information
df['div'] = df.apply(getdivinfo,axis=1,df_og_=df_og)
df = df[df['div'].apply(lambda x: len(x)>0)]
df['div'] = df['div'].apply(lambda x: x[0])
# add idx glob information
df['idx_glob'] = df.apply(getidxinfo,axis=1,df_og_=df_og)
df = df[df['idx_glob'].apply(lambda x: len(x)>0)]
df['idx_glob'] = df['idx_glob'].apply(lambda x: x[0])

## split each species dataframe individually
df_train, df_valid = splitTrainValid_cell(df)
## further split each into div and nondiv
df_div_train = df_train[df_train['div']==1]
df_nondiv_train = df_train[df_train['div']==0]
df_div_valid = df_valid[df_valid['div']==1]
df_nondiv_valid = df_valid[df_valid['div']==0]

## create dataset object
trainset_div = YeastBuddingDataset(df_div_train)
trainset_nondiv = YeastBuddingDataset(df_nondiv_train)
validset_div = YeastBuddingDataset(df_div_valid)
validset_nondiv = YeastBuddingDataset(df_nondiv_valid)

## create dataloaders
trainloader_div = DataLoader(trainset_div, batch_size=batch_num, shuffle=True)
validloader_div = DataLoader(validset_div, batch_size=batch_num*10, shuffle=True)
trainloader_nondiv = DataLoader(trainset_nondiv, batch_size=batch_num, shuffle=True)
validloader_nondiv = DataLoader(validset_nondiv, batch_size=batch_num*10, shuffle=True)

## create device variable to load model onto gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(comment='MAEDIV_equal_pos_neg')

##### MODEL INSTANTIATION #################

celldiv_mae = CELLDIV_MAE(patch_size=int(config['model']['patch_size']),
                          in_chan=int(config['model']['in_chan']),
                          n_enc_layers=int(config['model']['n_enc_layers']),
                          n_heads_enc=int(config['model']['n_heads_enc']),
                          n_layers_frame=int(config['model']['n_layers_frame']),
                          n_heads=int(config['model']['n_heads']),
                          init_enc_path=mae_init_enc_path,
                          pos_emb_path=mae_pos_emb_path,
                          mae_enc_path=mae_enc_path).to(device)


celldiv_mae_EMA = CELLDIV_MAE(patch_size=int(config['model']['patch_size']),
                          in_chan=int(config['model']['in_chan']),
                          n_enc_layers=int(config['model']['n_enc_layers']),
                          n_heads_enc=int(config['model']['n_heads_enc']),
                          n_layers_frame=int(config['model']['n_layers_frame']),
                          n_heads=int(config['model']['n_heads']),
                          init_enc_path=mae_init_enc_path,
                          pos_emb_path=mae_pos_emb_path,
                          mae_enc_path=mae_enc_path).to(device)

#### SET UP EMA ################

# load preload weight if pretrain
if pretrained_weight != 'None':
    celldiv_mae.load_state_dict(pretrained_weight)
# copy weights from original model to EMA model
celldiv_mae_EMA.load_state_dict(celldiv_mae.state_dict())

#### SET UP SCHEDULER ##########

# set up optimizer
optimizer = torch.optim.AdamW(celldiv_mae.parameters(),lr=lr,
                                         weight_decay=wdecay)

# create warmup schedulers
warmup_sch = Warmup_sch(optimizer,warm_steps,lr)

# create cosine annealing schedulers
cos_sch = CosineAnnealingWarmRestarts(optimizer,
                                      T_0=cos_total_steps,
                                      T_mult=T_mult)


##### START TRAINING ##########################

# create iter objects
data_iter_div_train = iter(trainloader_div)
data_iter_div_valid = iter(validloader_div)

data_iter_nondiv_train = iter(trainloader_nondiv)
data_iter_nondiv_valid = iter(validloader_nondiv)

# get batch number
batch_number_div_train = len(trainloader_div)
batch_number_div_valid = len(validloader_div)

batch_number_nondiv_train = len(trainloader_nondiv)
batch_number_nondiv_valid = len(validloader_nondiv)


# set the number of accumulation steps
acc_step_num = batch_number_nondiv_train//accumulation_steps

# start training routine
count = 0 # keep track of actual number of total iteration
acc_count = 0 # keep track of number of model update iteration
for i in range(num_epoch):
    
    running_loss = 0.0
    
    # train loop    
    for step in range(acc_step_num):
        
        # zero optimizer
        optimizer.zero_grad()
        
        # zero running loss
        running_loss = 0

        # configure finetuning if needed
        if fine_tune == 'True':
            # freeze all layers except last one
            for pname,param in celldiv_mae.named_parameters():
                if 'out_ffw' not in pname:
                    param.requires_grad=False
            # set all modules to eval() except for the last one
            for mname,module in celldiv_mae.named_children():
                if mname == 'out_ffw':
                    module.train()
                else:
                    module.eval()
            
        # set model to train
        celldiv_mae.train()
        
        for substep in range(accumulation_steps): 
            # extract data: div and nondiv individually and then stitched together
            
            im_train_div,vecs_train_div = next(data_iter_div_train)
            im_train_div = im_train_div.float().to(device)
            vecs_train_div = vecs_train_div.to(device)

            im_train_nondiv,vecs_train_nondiv = next(data_iter_nondiv_train)
            im_train_nondiv = im_train_nondiv.float().to(device)
            vecs_train_nondiv = vecs_train_nondiv.to(device)

            im_train = torch.cat([im_train_div,im_train_nondiv],dim=0)
            vecs_train = torch.cat([vecs_train_div,vecs_train_nondiv],dim=0)
            ## check if div dataset runs out. If so, reset it:
            if (count+1)%batch_number_div_train == 0:
                data_iter_div_train = iter(trainloader_div)
            
            # get denoiser loss
            
            pred_train = celldiv_mae(im_train)
            loss = nn.CrossEntropyLoss()
            loss_train = loss(pred_train,torch.argmax(vecs_train,dim=1))

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
        EMA_updater(celldiv_mae,celldiv_mae_EMA,decay_val)
        
        
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
            if (acc_count+1) % batch_number_div_valid == 0:
                data_iter_div_valid = iter(validloader_div)
            if (acc_count+1) % batch_number_nondiv_valid == 0:
                data_iter_nondiv_valid = iter(validloader_nondiv)

            # extract data
            im_valid_div,vecs_valid_div = next(data_iter_div_valid)
            im_valid_div = im_valid_div.float().to(device)
            vecs_valid_div = vecs_valid_div.to(device)

            im_valid_nondiv,vecs_valid_nondiv = next(data_iter_nondiv_valid)
            im_valid_nondiv = im_valid_nondiv.float().to(device)
            vecs_valid_nondiv = vecs_valid_nondiv.to(device)

            im_valid = torch.cat([im_valid_div,im_valid_nondiv],dim=0)
            vecs_valid = torch.cat([vecs_valid_div,vecs_valid_nondiv],dim=0)
            
            ## get validation loss
            celldiv_mae_EMA.eval()

            pred_valid = celldiv_mae_EMA(im_valid)
            loss = nn.CrossEntropyLoss()
            loss_valid = loss(pred_valid,torch.argmax(vecs_valid,dim=1))

    
        # print result and save statistics
        if (acc_count+1)%int(config['training']['report_freq']) == 0:

            #turn the validation result into onehots
            f = nn.Softmax(dim=1)
            vv = to_np(torch.argmax(vecs_valid, dim=1))
            pv = to_np(torch.argmax(f(pred_valid),dim=1))
            # calculate confusion matrix
            Mv = confusion_matrix(y_true = np.reshape(vv,(-1)), y_pred = np.reshape(pv,(-1)))
            if len(Mv)>1:
                writer.add_scalar("valid_true_div_rate", float(Mv[0,0])/(Mv[0,0]+Mv[0,1]), step+1)
                writer.add_scalar("valid_true_nondiv_rate", float(Mv[1,1])/(Mv[1,0]+Mv[1,1]), step+1)
                writer.add_scalar("valid_sum_sum_rate", float(Mv[1,1])/(Mv[1,0]+Mv[1,1])+float(Mv[0,0])/(Mv[0,0]+Mv[0,1]), step+1)

            
            # print running loss training
            print('epoch: {} | step {} | running_loss: {}'.format(i,acc_count+1,running_loss))
            print('################')
            print('loss_valid: {} | step {}'.format(loss_valid.item(),acc_count+1,))
            print('valid_true_div_rate: {}'.format(float(Mv[0,0])/(Mv[0,0]+Mv[0,1])))
            print('valid_true_nondiv_rate: {}'.format(float(Mv[1,1])/(Mv[1,0]+Mv[1,1])))
            print('valid_sum_sum_rate: {}'.format(float(Mv[1,1])/(Mv[1,0]+Mv[1,1])+float(Mv[0,0])/(Mv[0,0]+Mv[0,1])))
            print('################')
            
            writer.add_scalar("train_loss", running_loss,acc_count+1)
            writer.add_scalar("valid_loss", loss_valid.item(),acc_count+1)

    
    #### reset training data iter
    data_iter_nondiv_train = iter(trainloader_nondiv)
    # save weights every n epoch
    if (i+1) % save_freq == 0: 
        torch.save(celldiv_mae_EMA.state_dict(),weights_path + 'MAEDIV_equal_pos_neg_epoch{}.pt'.format(i+1))
        # duplicate the model and save a torchscript version
        model_scripted = copy.deepcopy(celldiv_mae_EMA).to(torch.device(device_placed))
        model_scripted = torch.jit.script(model_scripted)
        model_scripted.save(weights_path + 'mae_div_torchscript.pt')

