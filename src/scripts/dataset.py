import os
import sys
import time
import copy
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision import transforms

class Galaxy10DECals(Dataset):
    """
    Loading Galaxy10 DECals dataset from .h5 file.
    
    Args:
        dataset_path (str): Path to h5 file.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, dataset_path: str, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        with h5py.File(self.dataset_path, "r") as f:
            self.img = f['images'][()]
            self.label = f['labels'][()]
            self.length = len(self.label)

    def __getitem__(self, idx):
        img = self.img[idx]
        label = torch.tensor(self.label[idx], dtype=torch.long)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return self.length


class Galaxy10DECalsTest(Dataset):
    """
    Loading Galaxy10 DECals test dataset from .h5 file.
    
    Test dataset has original images rotated at random angles.
    
    Args:
        dataset_path (str): Path to h5 file.
        custom_idxs (array-like, optional): Array of indices to select from the dataset.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, dataset_path: str, transform=None, custom_idxs=None):
        self.dataset_path = dataset_path
        self.transform = transform
        with h5py.File(self.dataset_path, "r") as f:
            self.img = f['images'][()]
            self.label = f['labels'][()]
            self.angle = f['angles'][()]
            self.redshift = f['redshifts'][()]

            if custom_idxs is not None:
                self.img = self.img[custom_idxs]
                self.label = self.label[custom_idxs]
                self.angle = self.angle[custom_idxs]
                self.redshift = self.redshift[custom_idxs]
                
            self.length = len(self.label)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.img[idx]
        label = torch.tensor(self.label[idx], dtype=torch.long)
        angle = torch.tensor(self.angle[idx], dtype=torch.float)
        redshift = torch.tensor(self.redshift[idx], dtype=torch.float)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label, angle, redshift
    
    
if __name__ == '__main__':
    
    dataset = Galaxy10DECals('/Users/snehpandya/Projects/GCNN_DA/data/Galaxy10_DECals.h5', transform=transforms.ToTensor())
    test_dataset = Galaxy10DECalsTest('/Users/snehpandya/Projects/GCNN_DA/data/test_data_imbalanced.hdf5', transform=transforms.ToTensor())
    
    print(f"Length of dataset: {len(dataset)}")
    print(f"Label: {dataset[0][1]}")
    print(f"Image shape: {dataset[0][0].shape}")
    print(f"Label: {dataset[1][1]}")
    print(f"Image shape: {dataset[1][0].shape}")
    
    print(f"Length of test dataset: {len(test_dataset)}")
    print(f"Label: {test_dataset[0][1]}")
    print(f"Angle: {test_dataset[0][2]}")
    print(f"Redshift: {test_dataset[0][3]}")
    print(f"Image shape: {test_dataset[0][0].shape}")
    
    