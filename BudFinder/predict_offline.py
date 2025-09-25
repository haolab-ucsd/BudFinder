import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
import tifffile as tiff
from io import StringIO
from csv import writer
import re
import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth
import torchvision.transforms as transforms
from Utils import to_np, process_frame, preprocess_centroid_df, stack, process_movie

"""
    Begin pipeline of cell division event classifier application - crop.py takes
    two inputs - 1. a tif file of all 6 traps 361 frames long, 2. a csv file of
    data on ONLY that tif file, to be gleaned by the user.
    
    Will return a dataframe of crops (of relevant cells) and data on these crops
    
    This df will be used as input to stack.py.
"""
    
def preprocessInputs(instack):
    resize= transforms.Resize((64,64))
    instack = torch.Tensor(instack) # convert to torch t
    instack = resize(instack)
    instack = (255-instack)/255.00
    return instack.unsqueeze(1)

def create_crop_df(movie, centroid_df: str) -> pd.DataFrame:
    crop_df = centroid_df[['tracks', 'Frame']].copy()
    crop_df['crop'] = centroid_df.apply(lambda row: process_frame(row, movie), axis=1)

    return crop_df

def offline_predict(stack_df: pd.DataFrame, model_loc: str) -> pd.DataFrame:
    # initiate model
    device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    celldiv_mae = torch.jit.load(model_loc)
    celldiv_mae.eval()
    # loop through each frame 
    for _, group in stack_df.groupby(by='tracks'):
        for i, row in group.iterrows():
            stack = np.stack(row['stack']) # necessary np.stack call or the model will not return properly
            # prediction
            imstack_processed = preprocessInputs(stack)
            imstack_processed = imstack_processed.float().to(device)
            out = celldiv_mae(imstack_processed.unsqueeze(0))
            result = to_np(torch.argmax(out)).item() # zero is division and one is non-division
            stack_df.at[i,'div'] = result
    return stack_df


def getCleanedList(listin):
        # function to clean div matrix to prevent overcounting of divisions
        listcleaned = []
        for idx,f in enumerate(listin[:-1]):
            last_rec = True
            # if next idx is not incremental, save frame
            if int(listin[idx+1])-int(f) > 1:
                if last_rec == False:
                    cond_list.append(f)
                    listcleaned.append(np.median(np.array(cond_list)))
                else:
                    listcleaned.append(f)
                last_rec = True
            else:
                if last_rec == True:
                    cont_list = []
                    last_rec = False
                cont_list.append(f)
                
        return listcleaned

def mv_pred_workflow(tif_movie_path: str, movie_num: int, centroid_df: pd.DataFrame, model_loc: str) -> pd.DataFrame:
    movie = tiff.imread(tif_movie_path)
    df = preprocess_centroid_df(centroid_df, movie_num) # preprocess the dataframe to fill in any missing frames
    crop_df = create_crop_df(movie, df) # CROP DATAFRAME CREATION
    stack_df = stack(crop_df) # STACK DATAFRAME CREATION
    stack_df = offline_predict(stack_df, model_loc) ## PREDICT - OFFLINE ##

    ## CLEAN DIVLISTS AND RETURN TRACK_DIV MAP ##
    
    cell_track_divs = []
    for cell in np.unique(stack_df['tracks']): # looping through every unique track man number
        track_divs = dict()
        subdf = stack_df[stack_df['tracks']==cell] # get only the rows corresponding to the cell
        subdf = subdf.sort_values(by=['Frame']) # order by frame
        sdf = subdf[subdf['div']==0] # take only rows corresponding to a division
        div_frames = list(sdf['Frame'])
        pred_divlist = getCleanedList(div_frames) # input these frames into get cleaned list func
        track_divs['track_man'] = int(cell)
        track_divs['num_divs'] = len(pred_divlist)
        track_divs['div_frames'] = pred_divlist
        cell_track_divs.append(track_divs)

    divs_df = pd.DataFrame(cell_track_divs)
    
    return divs_df

def full_pipeline(movie_folder_path, csv_path, model_loc) -> None:
    df = pd.read_csv(csv_path)
    movies = [file for file in os.listdir(movie_folder_path) if file.endswith('c1.tif')]

    movie_map = dict()
    for movie in movies:
        movie_path = os.path.join(movie_folder_path, movie) # construct filepath to this movie
        movie_num = int(re.search(r'\d+', movie).group()) # get first int in movie name (which is the movie number)
        print(f'movie_num: {movie_num}')
        movie_df = df.loc[df['Movie'] == movie_num] # select only the rows for this movie in divstack cleaned df
        divs_df = mv_pred_workflow(movie_path, movie_num, movie_df, model_loc)
        # add track_divs to an overarching dict
        movie_map[movie_num] = divs_df

    predict_df = pd.concat(movie_map, names=['movie']).reset_index(level=0, drop=False)
    predict_df.to_csv('div_predictions.csv', index=False)
    
def main():
    # example usage
    movie_folder_path = '/Users/adrianlayer/hao_lab/biostats_final/zhen_hap4oe_imstackalign'
    csv_path = '/Users/adrianlayer/hao_lab/biostats_final/csvs/CropProperties_processed_sizecut_genotype.csv'
    offline_model_path = './OfflineModels/mae_div_complete.pt'
    # run full pipeline
    full_pipeline(movie_folder_path, csv_path, offline_model_path)

if __name__ == '__main__':
    main()