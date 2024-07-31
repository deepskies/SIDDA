import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Callable, Optional
from torch.utils.data import DataLoader
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
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(180),
        transforms.Resize(100),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Normalize(mean=(0.5, ), std=(0.5,))
    ])
    shape_dataset = Shapes(input_path_shapes, output_path_shapes, transform=train_transform)
    batch_size = 32  # You can set this to any value you want

    # Wrap the dataset in a DataLoader
    data_loader = DataLoader(shape_dataset, batch_size=batch_size, shuffle=True)

    # Get a single batch from the DataLoader
    batch = next(iter(data_loader))

    # If your dataset returns both inputs and targets, you might need to unpack them:
    inputs, targets = batch
        # Get the shape of the inputs and targets
    input_shape = inputs.shape
    target_shape = targets.shape
    
    print(f"Input shape: {input_shape}")
    print(f"Target shape: {target_shape}")
    
