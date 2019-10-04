import os
from os import listdir
import torchvision
import cv2

from src.config.config import *
from src.config.wgan import *
from src.attributes import *
from fastai.vision import *


    
def save_pred(generator, path, num_gen):
    if (os.path.isdir(path)==False): os.mkdir(path)
    for i in range(num_gen):
        random_noise = torch.randn(1, 100, 1, 1, device="cuda:0")
        mask = generator(random_noise)
        torchvision.utils.save_image(mask[0][0], path + '/mask_gen_' + str(i) + ".jpg")
        
        
def image_cleaner(path):
    #lag imagelist
    imagelist = listdir(path)
    #iterer igjennom listen
    for i in range(len(imagelist)):
        full_img_path = path+ '/' + imagelist[i]
        
        img = get_img(full_img_path)
        os.remove(full_img_path)
        
        if(not_blank(img)):
            new_img = undesired_objects(img)
            torchvision.utils.save_image(torch.from_numpy(new_img), full_img_path)
            
    new_imagelist = listdir(path)
    name_fixer(path,new_imagelist, imagelist)
    
def get_img(img_path):
    img = cv2.imread(img_path)
    cv2.imshow('image', img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def not_blank(img):
    is_it_blank = False
    if(np.unique(img).size > 1):
        is_it_blank = True
    return is_it_blank

def undesired_objects(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    return img2

def name_fixer(path, new_list, old_list):
    
    for i in range(len(new_list)):
        old_path = path + '/' + old_list[i]
        new_path = path + '/' + new_list[i]
        os.rename(new_path, old_path)  
