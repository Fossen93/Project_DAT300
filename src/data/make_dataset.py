import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def split_train_test (path_img, path_mask, percent_test=0.2, is_validation=False, percent_validation=0.2):
    np.random.seed(42)
    
    test_data, train_data = choose_test(path_img, percent_test)
    test_data = test_data.rename(columns={"Filenames": "Image"})
    test_data['Mask'] = test_data['Image']

    for i in range (len(test_data['Mask'])):
        name = test_data['Mask'][i].split('.')[0] + '_segmentation.png'
        test_data['Mask'][i] = name

    train_data = train_data.rename(columns={"Filenames": "Image"})
    train_data['Mask'] = train_data['Image']

    for i in range (len(train_data['Mask'])):
        name = train_data['Mask'][i].split('.')[0] + '_segmentation.png'
        train_data['Mask'][i] = name
        
    if (is_validation):
        is_validation = np.random.choice(a=[True, False], size=len(train_data), p=[percent_validation, 1-percent_validation])
        train_data['is_validation'] = is_validation
        
    return train_data, test_data

def choose_test(path, percent_test):
    file_names = os.listdir(path)
    index = []
    num_data = int(round(len(file_names)*percent_test))
    
    #cleans the filenames
    for i in range (len(file_names)):
        if (file_names[i].startswith('ISIC')==False):
            index.append(i)            

    file_names = np.delete(file_names, index)        
    
    np.random.shuffle(file_names)
    
    test_df = pd.DataFrame()
    test_df['Image'] = file_names[:num_data]
    rest_df = pd.DataFrame()
    rest_df['Image'] = file_names[num_data:]

    return test_df, rest_df

def show_batch (df, path_img, path_mask = None):

    images = []
    for index, row in df.sample(n=4).iterrows(): #glob.glob('images/*.jpg'):
        images.append(mpimg.imread(path_img + '/' + row['Image']))
        if(path_mask != None): images.append(mpimg.imread(path_mask + '/' + row['Mask']))
    
    plt.figure(figsize=(20,10))
    columns = 4
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)