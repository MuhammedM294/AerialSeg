from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2
import os
import torch
import albumentations as A
IMAGE_SIZE = 512
df_path = 'data/roads/metadata.csv'
parent_dir = os.path.split(df_path)[0]

class SegmentationDataset(Dataset[any]):
    """
    A PyTorch dataset class for segmentation task.

    This class represents a dataset used for the segmentation task, where each sample consists of an image and its corresponding mask. 
    It inherits from the PyTorch `Dataset` class.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing the image and label paths.
        train (bool): Specifies whether the dataset is for training (True) or testing (False).
        transform (torchvision.transforms.Normalize, optional): A transformation to be applied to the image. Defaults to None.
        augment (albumentations.Compose, optional): A composition of augmentations to be applied to both the image and mask. Defaults to None.

    Attributes:
        train (bool): Indicates whether the dataset is for training or testing.
        transform (torchvision.transforms.Normalize): The transformation applied to the image.
        augment (albumentations.Compose): The composition of augmentations applied to the image and mask.
        df (pd.DataFrame): The subset of the DataFrame that corresponds to the train or test split.

    Methods:
        __len__():
            Returns the length of the dataset.

        __getitem__(idx:int):
            Retrieves the image and mask at the given index.

        get_image(idx:int):
            Read the image at the given index from the file.

        get_mask(idx:int):
            Read the mask at the given index from the file.

        preprocess(image:np.array, mask:np.array):
            Preprocesses the image and mask data.

    Returns:
        A PyTorch dataset object for the segmentation task.
    """

    def __init__(self, df:pd.DataFrame, train:bool, transform:transforms.Normalize = None, augment:A.Compose = None):
     
        self.train = train
        self.transform = transform
        self.augment = augment

        if self.train:
            self.df = df[df['split'] == 'train']
        else:
            self.df = df[df['split'] == 'test']

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx:int):
        image = self.get_image(idx)
        mask = self.get_mask(idx)
        image , mask  = self.preprocess(image, mask)
    
        return image, mask
    
    def get_image(self, idx:int):
        image_path = parent_dir+'/'+ self.df.iloc[idx, 4]
        image = cv2.imread(image_path)[:,:,::-1]
        image = cv2.resize(image/255., (IMAGE_SIZE, IMAGE_SIZE))
        return image
    
    def get_mask(self, idx:int):
        mask_path = parent_dir +'/'+ self.df.iloc[idx, 5]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask/255, (IMAGE_SIZE, IMAGE_SIZE))
        mask = np.expand_dims(mask, axis=-1)
        return mask
    
    def preprocess(self, image:np.array, mask:np.array):
        
        if self.augment:
            data = self.augment()(image = image, mask = mask)
            image = data['image']
            mask = data['mask']
        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)
        mask = torch.tensor(mask, dtype=torch.float32).permute(2,0,1)
        if self.transform:
            image = self.transform()(image)
        
        return image, mask


def train_augmentation():
    """
    Creates an augmentation pipeline for training images.

    Returns:
        A.Compose: An instance of the `A.Compose` class representing the augmentation pipeline.

    Example:
        augmentation = train_augmentation()(image=image, mask=mask)
        augmented_image = augmentation['image']
        augmented_mask = augmentation['mask']
    """
    return  A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                            border_mode=cv2.BORDER_REFLECT),
        
    ])

def transform():
    """
    Returns a transformation function that normalizes an input image.

    Returns:
        torchvision.transforms.Normalize: An instance of the `torchvision.transforms.Normalize` class.

    Example:
        transform_func = transform()
        transformed_image = transform_func(image)
    """
    return transforms.Normalize(mean=[0.485,0.456, 0.406],std=[0.229, 0.224, 0.225])


def create_dataloader(df:pd.DataFrame, train:bool, 
                      transform:transforms.Normalize = None, augment:A.Compose = None, 
                      batch_size:int = 32, shuffle:bool = True, drop_last:bool = True):
    """
    Creates a PyTorch dataloader for the segmentation dataset.

    This function creates a PyTorch dataloader for the segmentation dataset using the provided DataFrame and dataset parameters.

    Args:
        df (pd.DataFrame): A Pandas DataFrame containing information about the images and masks paths.
        train (bool): Specifies whether the dataset is for training (True) or testing (False).
        transform (torchvision.transforms.Normalize, optional): A transformation to be applied to the image. Defaults to None.
        augment (albumentations.Compose, optional): A composition of augmentations to be applied to both the image and mask. Defaults to None.
        batch_size (int, optional): The batch size for the dataloader. Defaults to 32.
        shuffle (bool, optional): Specifies whether to shuffle the data during training. Defaults to True.
        drop_last (bool, optional): Specifies whether to drop the last incomplete batch if the dataset size is not divisible by the batch size. Defaults to True.
   
     Returns:
        A PyTorch dataloader object for the segmentation task.
    """
    dataset = SegmentationDataset(df, train, transform, augment)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle , drop_last=drop_last)

    return dataloader