import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, input_shape, mode="train"):
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        #self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        #temp = '/train/A'
        #temp2 = '/train/B'
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/A/*.*"))
        self.files2 = sorted(glob.glob(os.path.join(root, mode) + "/B/*.*"))

    def __getitem__(self, index):

        #img = Image.open(self.files[index % len(self.files)])
        #w, h = img.size
        #img_A = img.crop((0, 0, w / 2, h))
        #img_B = img.crop((w / 2, 0, w, h))
        
        img_A = Image.open(self.files[index % len(self.files)])
        img_B = Image.open(self.files2[index % len(self.files)])

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)

    
# To get the generated data files
class GenDataset(Dataset):
    def __init__(self, root, input_shape):
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        
        self.root = root

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        
        img = Image.open(self.files[index % len(self.files)])

        if np.random.random() < 0.5:
            img = Image.fromarray(np.array(img)[:, ::-1, :], "RGB")

        img = self.transform(img)
        
        name = str(self.files[index].replace(self.root+"/",''))
        
        return img, name

    def __len__(self):
        return len(self.files)
    
    
class TestImageDataset(Dataset):
    def __init__(self, root, input_shape, mode="test"):
        self.transform = transforms.Compose(
            [
                transforms.Resize(input_shape[-2:], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

    def __getitem__(self, index):
        
        img_A = Image.open(self.files[index % len(self.files)])

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)

        return {"A": img_A}

    def __len__(self):
        return len(self.files)