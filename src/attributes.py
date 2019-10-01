import os

from src.config import *

def create_dir(path):
    if(os.path.isdir(path)==False): os.mkdir(path)
    