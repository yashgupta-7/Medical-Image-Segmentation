# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:48:45 2020

@author: Administrator
"""
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import os

class RetinaDataset(Dataset):
    """Retina STARE/DRIVE dataset."""

    def __init__(self, img_folder, labels_folder):
        """
        Args:
            img_folder (string): Directory with all the training images.
            labels_folder (string): Directory with all the image labels.
        """
        self.images = os.listdir(img_folder)
        self.img_folder = img_folder
        self.labels_folder = labels_folder
        
        self.tx = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        
        self.mx = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x : torch.cat([x,1-x], dim=0))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        i1 = Image.open(self.img_folder + self.images[i])
        m1 = Image.open(self.labels_folder + self.images[i])
        return self.tx(i1), self.mx(m1)