# transformer based model to predict cell division event: Each frame goes through an MAE encoder and then the token is added
# the output token for each frames goes to another transformer block for complete prediction

import torch
import numpy as np
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import random
import math
from torchvision.ops import StochasticDepth
import torchvision.transforms as transforms
import skimage.io as io
from sklearn.metrics import confusion_matrix
from PIL import Image
import tifffile as tiff
from io import StringIO
from csv import writer
from joblib import Parallel, delayed
import copy


# patchify function
def patchify(in_im_:torch.Tensor,patch_size:int):
    out_im_ = in_im_.view(in_im_.size(0),in_im_.size(1),in_im_.size(2)//patch_size,patch_size,in_im_.size(3))
    out_im_ = out_im_.permute(0,1,2,4,3).contiguous()
    out_im_ = out_im_.view(in_im_.size(0),in_im_.size(1),(in_im_.size(2)//patch_size)*(in_im_.size(3)//patch_size),patch_size*patch_size)
    out_im_ = out_im_.permute(0,2,3,1).contiguous()
    out_im_ = out_im_.view(in_im_.size(0),(in_im_.size(2)//patch_size)*(in_im_.size(3)//patch_size),patch_size*patch_size*in_im_.size(1))
    return out_im_
# depatchitfy function
def depatchify(in_im_:torch.Tensor,in_dim:int,patch_size:int,in_chan:int):
    out_im_ = in_im_.view(in_im_.size(0),in_im_.size(1),patch_size*patch_size,in_chan)
    out_im_ = out_im_.permute(0,3,1,2).contiguous()
    out_im_ = out_im_.view(in_im_.size(0),in_chan,in_dim//patch_size,in_dim//patch_size,patch_size,patch_size)
    out_im_ = out_im_.permute(0,1,2,4,3,5).contiguous()
    return out_im_.view(in_im_.size(0),in_chan,in_dim,in_dim)

# self attention block
class SelfAttentionBlock(nn.Module):
    def __init__(self,embed_dim,n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim,n_head,
                                          dropout=0.1,batch_first=True)
    def forward(self,x):
        x,_ = self.attn(x,x,x)
        return x

# transformer block
class TransformerBlock(nn.Module):
    def __init__(self,embed_dim,n_head):  
        super().__init__()
        self.sattn = SelfAttentionBlock(embed_dim,n_head)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffw = nn.Sequential(nn.Linear(embed_dim,embed_dim*2),
                                 nn.SiLU(),nn.LayerNorm(embed_dim*2),nn.Dropout(p=0.1),
                                 nn.Linear(embed_dim*2,embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)
    def forward(self,x):
        x_att = self.sattn(x)
        x = self.norm1(x+x_att)
        x = self.norm2(x+self.ffw(x))
        return x

# transformer sequence blocks
class TransformerSequence(nn.Module):
    def __init__(self,n_layers,embed_dim,n_head):
        super().__init__()
        # transformer layers
        self.trans = nn.ModuleList([])
        for i in range(n_layers):
            self.trans.append(TransformerBlock(embed_dim,n_head))
        
    def forward(self,x):
        for subtrans in self.trans:
            x = subtrans(x)
        return x

# sinusoidal embedding for frame
def SinusoidalEmbedding(t_in:torch.Tensor,embed_dim:int,dev:torch.device,n:int=10000):
    # convert raw time into tensor with the right shape
    t_in = t_in.unsqueeze(1).expand(-1,embed_dim).view(t_in.size(0),embed_dim//2,2).contiguous().float().to(dev)
    # calculate sinusoidal embedding
    d = torch.pow(n,2*torch.arange(embed_dim//2)/embed_dim).unsqueeze(0).to(dev)
    t_in[:,:,0] = torch.cos(t_in[:,:,0]/d)
    t_in[:,:,1] = torch.sin(t_in[:,:,1]/d)
    return t_in.view(t_in.size(0),-1).contiguous()

## MAE Model
class MAE_2d(nn.Module):
    def __init__(self,in_chan=1, # number of input channels
                      in_dim=64, # input size
                      patch_size=8, # patch size
                      n_enc_layers=4, # encoder depth
                      n_dec_layers=1, # decoder depth
                      mask_frac= 0.75, # fraction of mask patches
                      n_heads=4):
        super().__init__()
        self.in_chan = in_chan
        self.in_dim = in_dim
        self.patch_size = patch_size
        self.embed_dim = patch_size*patch_size*in_chan
        # caclulate number of unmasked patches
        self.enc_patch_num = int(((in_dim//patch_size)**2)*(1-mask_frac))
        # make position embedding
        self.pos_emb = nn.Parameter(torch.randn(1,(in_dim//patch_size)**2,self.embed_dim))
        # make mask embedding
        self.mask_emb = nn.Parameter(torch.randn(1,1,self.embed_dim))
        # encoder
        self.encoder = TransformerSequence(n_enc_layers,self.embed_dim,n_heads)
        # initial encoder
        self.init_enc = nn.Sequential(nn.Linear(self.embed_dim,self.embed_dim*2),
                                     nn.SiLU(),nn.LayerNorm(self.embed_dim*2),nn.Dropout(p=0.1),
                                     nn.Linear(self.embed_dim*2,self.embed_dim),
                                     nn.SiLU(),nn.LayerNorm(self.embed_dim))
        # decoder
        self.decoder = TransformerSequence(n_dec_layers,self.embed_dim,n_heads)
        # out decoder
        self.out_dec = nn.Linear(self.embed_dim,self.embed_dim)

    
    def forward(self,x):
        # patchify the input
        x_patches = patchify(x,self.patch_size)
        # get the shuffled indices and shuffle both the patches and the indices
        indices = torch.randperm(x_patches.size(1))
        x_patches_shuffled = x_patches[:,indices,:]
        x_pemb_shuffled = self.pos_emb[:,indices,:]
        # get the encoding parts
        x_patch_enc = x_patches_shuffled[:,:self.enc_patch_num,:]
        # get the mask parts
        x_mask_emb_used = self.mask_emb.expand(x.size(0),(self.in_dim//self.patch_size)**2-self.enc_patch_num,x_patches.size(2))
        x_mask_emb_used = x_mask_emb_used + x_pemb_shuffled[:,self.enc_patch_num:,:]
        # encoding of patches
        x_in = self.init_enc(x_patch_enc)+x_pemb_shuffled[:,:self.enc_patch_num,:] # add position embedding after init encoding
        x_in = self.encoder(x_in)
        # concatenate with mask patches
        x_in = torch.cat((x_in,x_mask_emb_used),dim=1)
        # push through decoder
        x_out = self.decoder(x_in)
        x_out = self.out_dec(x_out)
        # reorganize
        x_out = x_out[:,torch.argsort(indices),:]
        return depatchify(x_out,self.in_dim,self.patch_size,self.in_chan)


## CELDIV_MAE Model    
class CELLDIV_MAE(nn.Module):
    def __init__(self,patch_size=8,
                      in_chan=1,
                      n_enc_layers=4,
                      n_heads_enc=4,
                      n_layers_frame=5,
                      n_heads=4,
                      init_enc_path='MAE_init_enc.pt',
                      pos_emb_path='MAE_pos_emb.pt',
                      mae_enc_path='MAE_encoder.pth'):
        super().__init__()


        self.embed_dim = patch_size*patch_size*in_chan
        self.t_emb_dim = patch_size*patch_size*in_chan
        self.patch_size = patch_size
        # get the mae init encoding layer
        self.mae_init_enc = torch.load(init_enc_path)

        # get the mae encoder model
        self.mae_encoder = torch.load(mae_enc_path)

        # get the positional embedding for the mae encoder
        self.pos_emb = torch.load(pos_emb_path)
        
        # get the cls token for the MAE
        self.cls_emb_mae = nn.Parameter(torch.randn(1,1,self.embed_dim))
        # get the cls token for the frame trans
        self.cls_emb_frame = nn.Parameter(torch.randn(1,1,self.embed_dim))
        # get the frame predictor transformer
        self.frame_trans = TransformerSequence(n_layers_frame,self.embed_dim,n_heads)
        # output ffw
        self.out_ffw = nn.Sequential(nn.Linear(self.embed_dim,self.embed_dim//2),
                                     nn.SiLU(),nn.LayerNorm(self.embed_dim//2),nn.Dropout(p=0.1),
                                     nn.Linear(self.embed_dim//2,2))
    def forward(self,x):
        # input x has shape [batch, frame, channel, H, W]
        # create empty tensor for frame token input
        x_frame_tokens = torch.zeros(x.size(0),x.size(1),self.embed_dim,device=x.device)
        # create mae token from each frame
        for t in range(x.size(1)):
            x_frame = x[:,t,:,:,:] # b c h w
            # patchify and add pos emb
            x_frame_patches = patchify(x_frame,self.patch_size) + self.pos_emb
            # add cls token
            x_frame_patches = torch.cat((self.cls_emb_mae.expand(x_frame_patches.size(0),1,-1),x_frame_patches),dim=1)
            # push through the encoder and get the token output back
            x_frame_out = self.mae_encoder(x_frame_patches)[:,0,:]
            # add the result back to the token tensor
            x_frame_tokens[:,t,:] = x_frame_out
        # create frame pos embedding
        t_pos_emb = SinusoidalEmbedding(torch.arange(x_frame_tokens.size(1)),self.t_emb_dim,x.device).unsqueeze(0) # (1,11,t_emb_dim)
        # add position embedding
        x_frame_tokens = x_frame_tokens + t_pos_emb
        # add frame token
        x_frame_tokens = torch.cat((self.cls_emb_frame.expand(x_frame_tokens.size(0),1,-1),x_frame_tokens),dim=1)
        # push through the frame trans and get the toke back
        x_out = self.frame_trans(x_frame_tokens)[:,0,:]
        # push through out ffw
        return self.out_ffw(x_out)
