import os
import uuid
import pandas as pd
import shutil


from src.config.config import *

def setup_data_directory(model_id, model_type, directories):
    
    for i in directories:
        create_dir(path_gen + '/' + i)
        create_dir(path_gen + '/' + i + '/' + model_type)
        create_dir(path_gen + '/' + i + '/' + model_type + '/' + model_id)

    
def setup_models_directory(model_type, directories):
    
    for i in directories:
        create_dir(path_models + '/' + i)
        create_dir(path_models + '/' + i + '/' + model_type)
        
def delete_model(model_id, model_type, data_dir, model_dir):
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
    if(os.path.isdir(path)==False): os.mkdir(path)
    
def create_model_id(models):
    
    id_found = False
    while (id_found is not True):
        id = uuid.uuid4().hex[:6]
        id_found = (id in models['Id']) is False
            
    return id

def add_model_stats(model_stats, models, overwrite=False):
    #models = pd.read_csv(path_models + '/model_stats.csv')
    if model_stats['Id'].values[0] in models['Id'].values:
        if overwrite:
            models = models[models.Id != model_stats['Id'].values[0]]
            models = models.append(model_stats, ignore_index = True)
        else:
            print('overwrite=False. Set overwrite to True if you want to delete old model')
    else:
        models = models.append(model_stats, ignore_index = True)

    return models
    #models.to_csv(path_models + '/model_stats.csv', index = False)
        