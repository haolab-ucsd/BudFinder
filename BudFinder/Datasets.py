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


########### CROP DATASET CLASS DEFINITION ##################################################
resize= transforms.Resize((64,64))
# make a torch dataset object that can return the image stack along with with its label for both train and valid
# image stack has the size (11,224,224) (frame,height,width)
class CellCropDataset(torch.utils.data.Dataset):
    def __init__(self,dataframe):
        self.dataframe = dataframe
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self,idx):
        mov = int(self.dataframe.iloc[idx,self.dataframe.columns.get_loc('movie')])
        fram = int(self.dataframe.iloc[idx,self.dataframe.columns.get_loc('Frame')])
        tr = int(self.dataframe.iloc[idx,self.dataframe.columns.get_loc('tracks')])
        
        imstack = io.imread('crops/movie{}/frame{}/cell{}.png'.format(mov,fram,tr))
        
        imstack = torch.Tensor(imstack) # convert to torch t
        imstack = resize(imstack)
        imstack = (255-imstack)/255.00
          
        return imstack.unsqueeze(0)

# make a torch dataset object that can return the image stack along with with its label for both train and valid
# image stack has the size (11,224,224) (frame,height,width)
class YeastBuddingDataset(torch.utils.data.Dataset):
    def __init__(self,dataframe):
        self.dataframe = dataframe
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self,idx):
        mov = int(self.dataframe.iloc[idx,self.dataframe.columns.get_loc('movie')])
        fram = int(self.dataframe.iloc[idx,self.dataframe.columns.get_loc('Frame')])
        tr = int(self.dataframe.iloc[idx,self.dataframe.columns.get_loc('tracks')])
        
        imstack = io.imread('stacks/movie{}/frame{}/stack_cell{}.tif'.format(mov,fram,tr))
        
        imstack = torch.Tensor(imstack) # convert to torch t
        imstack = resize(imstack)
        imstack = (255-imstack)/255.00
          
        label = self.dataframe.iloc[idx,self.dataframe.columns.get_loc('div')]
        if label == 1:
            classvec = torch.Tensor(np.array([1,0]))
        elif label == 0:
            classvec = torch.Tensor(np.array([0,1]))
        return (imstack.unsqueeze(1),classvec)
