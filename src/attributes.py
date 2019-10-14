import os
import uuid
import pandas as pd
import shutil


from src.config.config import *

def setup_data_directory(model_id, model_type, directories):
    """Creates the directory for where to save the generated data

    Parameters
    ----------
    model_id: str
        the id of the model
    model_type: str
        the type of the model
    directories: list
        the directory names for what type of generated data 
    """
    for i in directories:
        create_dir(path_gen + '/' + i)
        create_dir(path_gen + '/' + i + '/' + model_type)
        create_dir(path_gen + '/' + i + '/' + model_type + '/' + model_id)

    
def setup_models_directory(model_type, directories):
    """Creates the directory for where to save the generated models

    Parameters
    ----------
    model_type: str
        the type of the model
    directories: list
        the directory names for which models to save
    """
    for i in directories:
        create_dir(path_models + '/' + i)
        create_dir(path_models + '/' + i + '/' + model_type)
        
def delete_model(model_id, model_type, data_dir, model_dir):
    """Deletes a saved model

    Parameters
    ----------
    model_id: str
        the id of the model
    model_type: str
        the type of the model
    data_dir: list
        the directories that the model have generated data
    model_dir: list
        the directories of where the models are saved
    """
    models = pd.read_csv(path_models + '/model_stats.csv')
    models = models[models.Id != model_id]
    models.to_csv(path_models + '/model_stats.csv', index = False)
    
    for i in data_dir:
        
        path = path_gen + '/' + i + '/' + model_type + '/' + model_id
        print(os.listdir(path))
        if(os.path.isdir(path)): 
            print(path)
            shutil.rmtree(path)
        
    for i in model_dir:
        path = path_models + '/' + i + '/' + model_type + '/' + model_id + '.pth'
        if(os.path.isfile(path)): os.remove(path)

def create_dir(path):
    """Creates the directory of a path

    Parameters
    ----------
    path: str
        the path + directory thats gonna be created
    """
    if(os.path.isdir(path)==False): os.mkdir(path)
    
def create_model_id(models):
    """ Creates an unique id for your model

    Parameters
    ----------
    models: pd.Dataframe
        containing information about all the saved models including their id's

    Returns
    -------
    str
        the unique id
    """
    id_found = False
    while (id_found is not True):
        id = uuid.uuid4().hex[:6]
        id_found = (id in models['Id']) is False
            
    return id

def add_model_stats(model_stats, models, overwrite=False):
    """ Adds information from a new model to the model_stats.csv file

    Parameters
    ----------
    model_stats: pd.DataFrame
        the new model thats going to be added to det models
    models: df
        the model_stats file thats going to be updated
    overwrite=False: bool
        If True you overwrite the model with the same id if it's already exists, else nothing happens

    Returns
    -------
    pd.DataFrame
        the updatet model_stats file as a pd.DataFrame
    """
    if model_stats['Id'].values[0] in models['Id'].values:
        if overwrite:
            models = models[models.Id != model_stats['Id'].values[0]]
            models = models.append(model_stats, ignore_index = True)
        else:
            print('overwrite=False. Set overwrite to True if you want to delete old model')
    else:
        models = models.append(model_stats, ignore_index = True)

    models.to_csv(path_models + '/model_stats.csv', index = False)
    return models 
        
def show_batch (df, path_img, path_mask=None):
    """ Shows a batch of images

    df: pd.DataFrame
        contains filenames of images and masks
    path_img: str 
        the path to the images
    path_mask=None: str
        the path to the masks. None if you don't want to display them
    """
    images = []
    for index, row in df.sample(n=4).iterrows(): #glob.glob('images/*.jpg'):
        images.append(mpimg.imread(path_img + '/' + row['Image']))
        if(path_mask != None): images.append(mpimg.imread(path_mask + '/' + row['Mask']))
    
    plt.figure(figsize=(20,10))
    columns = 4
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image)