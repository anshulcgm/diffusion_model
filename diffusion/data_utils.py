import pdb

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from torch.utils.data import Dataset, DataLoader


TRAIN_DIR = "archive/car_data/car_data/train/"

def load_dataset(train_dir: str) -> Dataset:
    """Creating the dataset for the train images
    
    Args:
        train_dir: Directory where all the images are located
    
    Returns:
        dataset: Dataset object with images all sized to resolution 224x224
    """
    image_transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    dataset = datasets.ImageFolder(train_dir, transform = image_transform)
    return dataset

def load_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    """Creating dataloader to run through the dataset
    
    Args:
        dataset: Dataset object containing images and labels for cars dataset
        batch_size: How many images to put in one batch
    
    Return:
        dataloader: Dataloader object for inputted dataset
    """
    dataloader = DataLoader(dataset, batch_size = batch_size)
    return dataloader

def prep_data(train_dir: str, batch_size: int) -> DataLoader:
    """Simple method to prep the cars dataset and return the dataloader

    Args:
        train_dir: Directory where the images are located
        batch_size: How many images in one batch
    
    Returns:
        dataloader: Dataloader with images all sized to 224x224 resolution
    """
    dataset = load_dataset(train_dir)
    dataloader = load_dataloader(dataset = dataset, batch_size = batch_size)
    return dataloader
