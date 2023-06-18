from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2
import torch
import albumentations as A
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_size = 512

class SegmentationDataset(Dataset[any]):

    def __init__(self, df:pd.DataFrame, train:bool, transform:bool , augment:bool, augementation):
        self.train = train
        if self.train:
            self.df = df[df['split'] == 'train']
        else:
            self.df = df[df['split'] == 'test']

        self.transform = transforms.Normalize(mean=[0.485,0.456, 0.406],std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx:int):
        image = self.get_image(idx)
        label = self.get_label(idx)
    
        return image.to(device) , label.to(device)
    
    def get_image(self, idx:int):
        self.idx = idx
        image = self.preprocess_image(self.idx)
        return image
    
    def get_label(self, idx:int):
        self.idx = idx
        label = self.preprocess_label(self.idx)
        return label
    
    def preprocess_image(self, idx:int):
        image_path = '../data/roads/' + self.df.iloc[idx, 4]
        image = cv2.imread(image_path)[:,:,::-1]
        image = cv2.resize(image, (image_size, image_size))
        image = torch.tensor(image/255, dtype=torch.float32).permute(2,0,1)
        if self.transform:
            image = self.transform(image)
        
        return image
    
    def preprocess_label(self, idx:int):
        label_path = '../data/roads/' + self.df.iloc[idx, 5]
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (image_size, image_size))
        label = np.expand_dims(label, axis=-1)
        label = torch.tensor(label/255, dtype=torch.float32)
        return label

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
        A.Resize(image_size, image_size),
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