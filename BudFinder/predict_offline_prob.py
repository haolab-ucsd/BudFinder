import os
import numpy as np
import pandas as pd
from PIL import Image
import tifffile as tiff
from io import StringIO
from csv import writer
import re
import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth
import torchvision.transforms as transforms

"""
    Begin pipeline of cell division event classifier application
"""

def to_np(x):
    return x.data.cpu().numpy()
    
def preprocessInputs(instack):
    resize= transforms.Resize((64,64))
    instack = torch.Tensor(instack) # convert to torch t
    instack = resize(instack)
    instack = (255-instack)/255.00
    return instack.unsqueeze(1)

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

def prep_divstack_cleaned(df: pd.DataFrame) -> pd.DataFrame:
    # in this case df is the divstack cleaned dataframe containing rows only from a single movie
    
    # only take necessary columns
    necessary_cols = ['Frame', 'tracks', 'Centroid_X', 'Centroid_Y']
    df = df[df.columns.intersection(necessary_cols)]
    df.sort_values(by='Frame').groupby(by='tracks')

    # need to fill in the frame holes in our divstack cleaned df before cropping
    # seek out all the missing frames and fill them in
    # write them into a csv memory object and then read that csv back into df
    # https://stackoverflow.com/questions/41888080/efficient-way-to-add-rows-to-dataframe

    output = StringIO()
    csv_writer = writer(output)
    csv_writer.writerow(df.columns)
    
    for _, group in df.groupby(by='tracks'):
        first_row = True
        for _, row in group.iterrows():
            if first_row: # base case
                prev_row = row
                prev_frame = row['Frame']
                csv_writer.writerow(row)
                first_row = False
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

def create_crop_df(movie, divstack_cleaned_df: str) -> pd.DataFrame:
    crop_df = divstack_cleaned_df[['tracks', 'Frame']].copy()
    crop_df['crop'] = divstack_cleaned_df.apply(lambda row: process_frame(row, movie), axis=1)

    return crop_df

def create_stack_df(crop_df: pd.DataFrame) -> pd.DataFrame:
    # insert 11 more columns into dataframe, corresponding to the stack frames
    
    grouped = crop_df.groupby(by='tracks')
    
    # fill out the needed 11 columns of offset crops
    for i in range(-5, 6, 1):
        if i < 0:
            crop_df[f'{i}'] = grouped['crop'].shift(-i).bfill() #fillna(method='bfill')
        else:
            crop_df[f'{i}'] = grouped['crop'].shift(-i).ffill() #fillna(method='ffill')

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
            prob = to_np(torch.softmax(out, dim=1)[0,0]).item() ###### ins
            stack_df.at[i,'div'] = result
            stack_df.at[i,'prob'] = prob ###### ins
    return stack_df

def mv_pred_workflow(tif_movie_path: str, divstack_cleaned_df: pd.DataFrame, model_loc: str) -> pd.DataFrame:
    movie = tiff.imread(tif_movie_path)
    df = prep_divstack_cleaned(divstack_cleaned_df) # preprocess the dataframe to fill in any missing frames
    crop_df = create_crop_df(movie, df) # CROP DATAFRAME CREATION
    stack_df = create_stack_df(crop_df) # STACK DATAFRAME CREATION
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
        track_divs['div_probabilities'] = list(subdf['prob'])
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
        divs_df = mv_pred_workflow(movie_path, movie_df, model_loc)
        # add track_divs to an overarching dict
        movie_map[movie_num] = divs_df

    predict_df = pd.concat(movie_map, names=['movie']).reset_index(level=0, drop=False)
    predict_df.to_csv('div_prediction_probabilities.csv', index=False)
    
def main():
    # example usage
    movie_folder_path = '/Volumes/Aging/Hetian/CellSegSoftwareNew2/imstackAlign'
    csv_path = '/Volumes/Aging/Hetian/CellSegSoftwareNew2/csv/CellProperties_tracked_man_all.csv'
    offline_model_path = '/Users/adrianlayer/hao_lab/sam/CellEventDetector/OfflineModels/mae_div_complete.pt'
    # run full pipeline
    full_pipeline(movie_folder_path, csv_path, offline_model_path)

if __name__ == '__main__':
    main()
