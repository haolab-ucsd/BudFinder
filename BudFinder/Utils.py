import torch
import numpy as np
import torch.nn as nn
import os
import torch.nn.functional as F
import pandas as pd
import random
import math
import skimage.io as io
from PIL import Image
import tifffile as tiff
from io import StringIO
from csv import writer
from joblib import Parallel, delayed
import copy
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts

############## MISCELLANEOUS ###########################################

def to_np(x):
    return x.data.cpu().numpy()
    
def splitTrainValid(in_data,valid_frac=0.2,seed = 54):
    
    np.random.seed(seed) 
    idxlist = np.arange(len(in_data))
    idxlist_valid = np.random.choice(idxlist,int(len(in_data)*valid_frac),replace=False)
    idxlist_train = np.setdiff1d(idxlist,idxlist_valid)
    
    return in_data.iloc[idxlist_train,:],in_data.iloc[idxlist_valid,:]

def splitTrainValid_cell(in_data,valid_frac=0.2,seed = 54):
    
    cellidx = np.unique(in_data['idx_glob'])
    random.shuffle(cellidx)
    
    
    idx_valid = cellidx[:int(valid_frac*len(cellidx))]
    idx_train = cellidx[int(valid_frac*len(cellidx)):]
    
    return in_data[in_data['idx_glob'].isin(idx_train)],in_data[in_data['idx_glob'].isin(idx_valid)]

# define EMA update function
def EMA_updater(model_now,model_running,decay_val):
    # get parameters for each model
    params_now = dict(model_now.named_parameters())
    params_running = dict(model_running.named_parameters())
    # loop through each model's named params to update it
    for name in params_now.keys():
        weight_now = params_now[name].data
        weight_running = params_running[name].data
        weight_running_new = weight_running*decay_val + weight_now*(1-decay_val)
        params_running[name].data.copy_(weight_running_new)

### CROP AND STACKS MAKING FUNCTIONS ##########################################
def crop_image(frame, x, y, crop_size=50):
    half_size = crop_size // 2
    height, width = frame.shape

    # Calculate crop boundaries ensuring they don't exceed image dimensions
    x_min = max(0, x - half_size)
    x_max = min(width, x + half_size)
    y_min = max(0, y - half_size)
    y_max = min(height, y + half_size)
    
    cropped_frame = frame[y_min:y_max, x_min:x_max]

    # Adjust the crop size to maintain the crop_size if needed
    if cropped_frame.shape[0] != crop_size or cropped_frame.shape[1] != crop_size:
        pad_y = (crop_size - cropped_frame.shape[0], 0) if y_max >= height else (0, crop_size - cropped_frame.shape[0])
        pad_x = (crop_size - cropped_frame.shape[1], 0) if x_max >= width else (0, crop_size - cropped_frame.shape[1])
        cropped_frame = np.pad(cropped_frame, (pad_y, pad_x), mode='constant', constant_values=0)
    
    return cropped_frame

def resize_image(image, new_size=(64, 64)):
    # Rescale the image to 0-255 range if it is not in np.uint8 format
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min()) * 255.0
        image = image.astype(np.uint8)
    
    resized_image = np.array(Image.fromarray(image).resize(new_size))
    return resized_image

def process_frame(row, movie):
    frame_number = int(row['Frame'])
    x = int(row['Centroid_X'])
    y = int(row['Centroid_Y'])
    
    frame = movie[frame_number - 1] # 0 indexed, not 1 indexed.
    cropped_image = crop_image(frame, x, y)
    resized_image = resize_image(cropped_image)
    
    return resized_image

def preprocess_centroid_df(df,movieid):
    df = df[df['Movie']==movieid]
    df.reset_index(inplace=True)
    # take only the columns we need from the divstack csv
    necessary_cols = ['Frame', 'tracks', 'Centroid_X', 'Centroid_Y']
    if not all(col in df.columns for col in necessary_cols):
        raise ValueError(f"CSV file must contain the following columns: {necessary_cols}")
    df = df[df.columns.intersection(necessary_cols)]
    df.sort_values(by='Frame').groupby(by='tracks')
    
    # rewrite the centroid dataframe, filling in missing frames by duping centroids
    # write into a csv memory object and then read that csv back into df
    # https://stackoverflow.com/questions/41888080/efficient-way-to-add-rows-to-dataframe
    output = StringIO()
    csv_writer = writer(output)
    csv_writer.writerow(df.columns)
    
    for _, group in df.groupby(by='tracks'):
        if len(group) < 6: # deals with the edge case where stack function can't handle groups of len 5 or less
            continue
        prev_frame = None
        for i, row in group.iterrows():
            if not prev_frame and row.equals(group.iloc[0]): # base case - check if prev_frame is None first so we don't do this check every time
                prev_row = row
                prev_frame = row['Frame']
                csv_writer.writerow(row)
                continue
            if row['Frame'] != (prev_frame + 1): # missing row
                diff = int(row['Frame'] - (prev_frame + 1))
                for _ in range(diff):
                    prev_row['Frame'] += 1
                    csv_writer.writerow(prev_row)
            prev_row = row
            prev_frame = row['Frame']
            csv_writer.writerow(row)
            
    output.seek(0) # we need to get back to the start of the BytesIO
    df = pd.read_csv(output)
    
    return df

def process_movie(csv_path, tif_path, movieid):
    
    # check for tif file existence
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f'tif file {tif_path} does not exist.')
    
    movie = tiff.imread(tif_path)
    df = preprocess_centroid_df(csv_path,movieid)
    
    crop_df = df[['tracks', 'Frame']].copy()

    # edge case: no viable rows in original csv for given movie - skip
    if not df.empty:
        crop_df['crop'] = df.apply(lambda row: process_frame(row, movie), axis=1)
    else:
        crop_df['crop'] = pd.Series(dtype='O')

    return crop_df

########## STACKS ##########
def stack(crop_df):
    # insert 11 more columns into dataframe, corresponding to the stack frames
    # considerations: edge cases (beg/end frames)

    # grouped object for access
    grouped = crop_df.groupby(by='tracks')

    # fill out the needed 11 columns of offset crops
    for i in range(-5, 6, 1):
        if i < 0:
            crop_df[f'{i}'] = grouped['crop'].shift(-i).bfill()
        else:
            crop_df[f'{i}'] = grouped['crop'].shift(-i).ffill()

    # get label columns
    stack_df = crop_df[['tracks', 'Frame']].copy()

    # get str list of crop columns
    crop_range = [*range(-5, 6, 1)]
    crop_range = [str(i) for i in crop_range]

    # grab these crop columns and put them in a numpy array
    crops = crop_df[crop_df.columns.intersection(crop_range)].to_numpy()

    # stack the crops
    stacks = np.stack(crops, axis=0)

    # put stacks into stack_df
    stack_df['stack'] = list(stacks)
    
    return stack_df

##########################
def savecrops(row):
    os.makedirs('crops/movie{}/frame{}'.format(int(row['movie']),
                                                int(row['Frame'])),exist_ok=True)
    io.imsave('crops/movie{}/frame{}/cell{}.png'.format(int(row['movie']),
                                                                  int(row['Frame']),
                                                                  int(row['tracks'])), np.stack(row['crop']))
    return
    
def savemovie_crop(fname,tifpath,csvpath):
    movieid = int(fname[fname.find('y')+1:fname.find('c')])
    dfid = process_movie(csvpath,tifpath+'/'+fname,movieid)
    dfid['movie'] = [movieid]*len(dfid)
    dfid.apply(savecrops,axis=1)
    # remove the stack column
    dfid.drop('crop',axis=1,inplace=True)
    return dfid



def savestacks(row):
    os.makedirs('stacks/movie{}/frame{}'.format(int(row['movie']),
                                                int(row['Frame'])),exist_ok=True)
    tiff.imwrite('stacks/movie{}/frame{}/stack_cell{}.tif'.format(int(row['movie']),
                                                                  int(row['Frame']),
                                                                  int(row['tracks'])), np.stack(row['stack']))
    return

# save tif stack into a movie and remove stack from stack df to save memory
def savemovie(fname,tifpath,csvpath):
    movieid = int(fname[fname.find('y')+1:fname.find('c')])
    crop_df = process_movie(csvpath,tifpath+'/'+fname,movieid)
    # edge case: no viable rows in original csv for given movie - skip
    if crop_df.empty:
        return
    dfid = stack(crop_df)
    dfid['movie'] = [movieid]*len(dfid)
    dfid.apply(savestacks,axis=1)
    # remove the stack column
    dfid.drop('stack',axis=1,inplace=True)
    return dfid

def getdivinfo(row,df_og_):
    mov_ = int(row['movie'])
    fram_ = int(row['Frame'])
    tr_ = int(row['tracks'])
    subdf_ = df_og_[(df_og_['Movie']==mov_) & (df_og_['Frame']==fram_) & (df_og_['track_man']==tr_)]
    return list(subdf_['div'])

def getidxinfo(row,df_og_):
    mov_ = int(row['movie'])
    fram_ = int(row['Frame'])
    tr_ = int(row['tracks'])
    subdf_ = df_og_[(df_og_['Movie']==mov_) & (df_og_['Frame']==fram_) & (df_og_['track_man']==tr_)]
    return list(subdf_['idx_glob'])
    
####### Schedulers #############
# create a warmup scheduler function
def Warmup_sch(optimizer,warmup_stepnum,target_lr):
    def lambda_lr(step_):
        if step_ < warmup_stepnum:
            return (float(step_)/warmup_stepnum)*target_lr
        else:
            return target_lr
    return LambdaLR(optimizer,lr_lambda=lambda_lr)

# define a function to combine warmup and cosine annealing
def combine_sch(step_,warm_steps,wsch, cos_sch):
    if step_ < warm_steps:
        wsch.step()
    else:
        cos_sch.step()
