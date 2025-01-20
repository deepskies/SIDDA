from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class Shapes(Dataset):
    """Dataset class for the shapes dataset.

    Args:
        input_path (str): Path to the input data.
        output_path (Optional[str], optional): Path to the output data. Defaults to None.
        transform (Optional[Callable], optional): Transform to apply to the data. Defaults to None.
        target_domain (bool, optional): Whether the dataset is used for target domain. Defaults to False.
    """

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_domain: bool = False,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.transform = transform
        self.target_domain = target_domain

        try:
            self.img = np.load(self.input_path)
            if not self.target_domain and self.output_path is not None:
                self.label = np.load(self.output_path)
        except Exception as e:
            raise RuntimeError(
                f"Error loading data from {input_path} and {output_path}: {e}"
            )

        if (
            not self.target_domain
            and self.output_path is not None
            and len(self.img) != len(self.label)
        ):
            raise ValueError("Input and output files must have the same length.")

        self.length = len(self.img)

    def __getitem__(self, idx: int):
        img = self.img[idx]
        if self.transform:
            img = self.transform(img)

        if self.target_domain:
            return img

        label = torch.tensor(self.label[idx], dtype=torch.long)
        return img, label

    def __len__(self) -> int:
        return self.length


class AstroObjects(Dataset):
    """Dataset class for the astronomical objects dataset.

    Args:
        input_path (str): Path to the input data.
        output_path (Optional[str], optional): Path to the output data. Defaults to None.
        transform (Optional[Callable], optional): Transform to apply to the data. Defaults to None.
        target_domain (bool, optional): Whether the dataset is used for target domain. Defaults to False.
    """

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_domain: bool = False,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.transform = transform
        self.target_domain = target_domain

        try:
            self.img = np.load(self.input_path)
            if not self.target_domain and self.output_path is not None:
                self.label = np.load(self.output_path)
        except Exception as e:
            raise RuntimeError(
                f"Error loading data from {input_path} and {output_path}: {e}"
            )

        if (
            not self.target_domain
            and self.output_path is not None
            and len(self.img) != len(self.label)
        ):
            raise ValueError("Input and output files must have the same length.")

        self.length = len(self.img)

    def __getitem__(self, idx: int):
        img = self.img[idx]
        if self.transform:
            img = self.transform(img)

        if self.target_domain:
            return img

        label = torch.tensor(self.label[idx], dtype=torch.long)
        return img, label

    def __len__(self) -> int:
        return self.length


class MnistM(Dataset):
    """Dataset class for the MNIST-M dataset.

    Args:
        input_path (str): Path to the input data.
        output_path (Optional[str], optional): Path to the output data. Defaults to None.
        transform (Optional[Callable], optional): Transform to apply to the data. Defaults to None.
        target_domain (bool, optional): Whether the dataset is used for target domain. Defaults to False.
    """

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_domain: bool = False,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.transform = transform
        self.target_domain = target_domain

        try:
            self.img = np.load(self.input_path)
            if not self.target_domain and self.output_path is not None:
                self.label = np.load(self.output_path)
        except Exception as e:
            raise RuntimeError(
                f"Error loading data from {input_path} and {output_path}: {e}"
            )

        if (
            not self.target_domain
            and self.output_path is not None
            and len(self.img) != len(self.label)
        ):
            raise ValueError("Input and output files must have the same length.")

        self.length = len(self.img)

    def __getitem__(self, idx: int):
        img = self.img[idx]
        if self.transform:
            img = self.transform(img)

        if self.target_domain:
            return img

        label = torch.tensor(self.label[idx], dtype=torch.long)
        return img, label

    def __len__(self) -> int:
        return self.length


class GZEvo(Dataset):
    """Dataset class for the Galaxy Zoo Evolution dataset.

    Args:
        input_path (str): Path to the input data.
        output_path (Optional[str], optional): Path to the output data. Defaults to None.
        transform (Optional[Callable], optional): Transform to apply to the data. Defaults to None.
        target_domain (bool, optional): Whether the dataset is used for target domain. Defaults to False.
    """

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_domain: bool = False,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.transform = transform
        self.target_domain = target_domain

        try:
            self.img = np.load(self.input_path)
            if not self.target_domain and self.output_path is not None:
                self.label = np.load(self.output_path)
        except Exception as e:
            raise RuntimeError(
                f"Error loading data from {input_path} and {output_path}: {e}"
            )

        if (
            not self.target_domain
            and self.output_path is not None
            and len(self.img) != len(self.label)
        ):
            raise ValueError("Input and output files must have the same length.")

        self.length = len(self.img)

    def __getitem__(self, idx: int):
        img = self.img[idx]
        if self.transform:
            img = self.transform(img)

        if self.target_domain:
            return img

        label = torch.tensor(self.label[idx], dtype=torch.long)
        return img, label

    def __len__(self) -> int:
        return self.length


dataset_dict = {
    "shapes": Shapes,
    "astro_objects": AstroObjects,
    "mnist_m": MnistM,
    "gz_evo": GZEvo,
}

gz_evo_classes = (
    "barred_spiral",
    "edge_on_disk",
    "featured_without_bar_or_spiral",
    "smooth_cigar",
    "smooth_round",
    "unbarred_spiral",
)
mnist_m_classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
shapes_classes = ("line", "rectangle", "circle")
astro_objects_classes = ("elliptical", "spiral", "stars")

classes_dict = {
    "shapes": shapes_classes,
    "astro_objects": astro_objects_classes,
    "mnist_m": mnist_m_classes,
    "gz_evo": gz_evo_classes,
}

if __name__ == "__main__":
    input_path_shapes = "/Users/snehpandya/Projects/GCNN_DA/data/toy_dataset_shapes/shapes_data_noisy.npy"
    output_path_shapes = "/Users/snehpandya/Projects/GCNN_DA/data/toy_dataset_shapes/shapes_labels_noisy.npy"

    input_path_blob = (
        "/Users/snehpandya/Projects/GCNN_DA/data/toy_dataset_blob/blob_data_noisy.npy"
    )
    output_path_blob = (
        "/Users/snehpandya/Projects/GCNN_DA/data/toy_dataset_blob/blob_labels_noisy.npy"
    )
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomRotation(180),
            transforms.Resize(100),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
    shape_dataset = Shapes(
        input_path_shapes, output_path_shapes, transform=train_transform
    )
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
