import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Callable, Optional

class Shapes(Dataset):
    def __init__(self, input_path: str, output_path: str, transform: Optional[Callable] = None):
        self.input_path = input_path
        self.output_path = output_path
        self.transform = transform
        
        try:
            self.img = np.load(self.input_path)
            self.label = np.load(self.output_path)
        except Exception as e:
            raise RuntimeError(f"Error loading data from {input_path} and {output_path}: {e}")
        
        if len(self.img) != len(self.label):
            raise ValueError("Input and output files must have the same length.")
        
        self.length = len(self.label)

    def __getitem__(self, idx: int):
        img = self.img[idx]
        label = torch.tensor(self.label[idx], dtype=torch.long)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

    def __len__(self) -> int:
        return self.length
    
    
class Blobs(Dataset):
    def __init__(self, input_path: str, output_path: str, transform: Optional[Callable] = None):
        self.input_path = input_path
        self.output_path = output_path
        self.transform = transform
        
        try:
            self.img = np.load(self.input_path)
            self.label = np.load(self.output_path)
        except Exception as e:
            raise RuntimeError(f"Error loading data from {input_path} and {output_path}: {e}")
        
        if len(self.img) != len(self.label):
            raise ValueError("Input and output files must have the same length.")
        
        self.length = len(self.label)

    def __getitem__(self, idx: int):
        img = self.img[idx]
        label = torch.tensor(self.label[idx], dtype=torch.long)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

    def __len__(self) -> int:
        return self.length
    
if __name__ == '__main__':
    
    input_path_shapes = '/Users/snehpandya/Projects/GCNN_DA/data/toy_dataset_shapes/shapes_data_noisy.npy'
    output_path_shapes = '/Users/snehpandya/Projects/GCNN_DA/data/toy_dataset_shapes/shapes_labels_noisy.npy'
    
    input_path_blob = '/Users/snehpandya/Projects/GCNN_DA/data/toy_dataset_blob/blob_data_noisy.npy'
    output_path_blob = '/Users/snehpandya/Projects/GCNN_DA/data/toy_dataset_blob/blob_labels_noisy.npy'
    
    shape_dataset = Shapes(input_path_shapes, output_path_shapes, transform=transforms.ToTensor())
    blob_dataset = Blobs(input_path_blob, output_path_blob, transform=transforms.ToTensor())
    
    plt.imshow(shape_dataset[0][0].permute(1, 2, 0))
    plt.show()
    plt.imshow(blob_dataset[0][0].permute(1, 2, 0))
    plt.show()
    