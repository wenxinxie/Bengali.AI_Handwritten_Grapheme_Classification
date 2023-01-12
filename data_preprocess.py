import numpy as np
import pandas as pd
import albumentations as A
import torch
from torchvision import transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy

import gc
import cv2

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
IMG_HEIGHT = 224
IMG_WIDTH = 224

NUM_GRAPHEME_ROOT = 168
NUM_VOWEL_DIACRITIC = 11
NUM_CONSONANT_DIACRITIC = 8

def load_info(paths):
    all_info = []
    for path in paths:
        info_df = pd.read_csv(path)
        all_info.append(info_df)
    all_info_df = pd.concat(all_info)
    return all_info_df

def load_images_np(paths):
    all_images = []
    for path in paths:
        image_df = pd.read_parquet(path)
        images = image_df.iloc[:, 1:].values.reshape(-1, 137, 236).astype(np.uint8)
        # all rows, columns excluding index
        del image_df
        gc.collect()
        all_images.append(images)
    all_images = np.concatenate(all_images)
    return all_images

def load_images_df(paths):
    df_images_list = []
    for path in paths:
        df_images = pd.read_parquet(path)
        df_images_list.append(df_images)
        del df_images
        gc.collect()
    images = pd.concat(df_images_list)
    return images

def image_df2np(image_df):
    image_np = image_df.iloc[:, 1:-1].values.reshape(-1, 137, 236).astype(np.uint8)
    return image_np


class Albumentations:
    def __init__(self, augmentations):
        self.augmentations = A.Compose(augmentations)

    def __call__(self, image):
        image = self.augmentations(image=image)['image']
        return image

preprocess = [
    A.CenterCrop(height=137, width=IMG_WIDTH),
    A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH, always_apply=True),
]

augmentations = [
    A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255], always_apply=True),
    A.imgaug.transforms.IAAAffine(shear=20, mode='constant', cval=255, always_apply=True),
    A.ShiftScaleRotate(rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255], mask_value=[255, 255, 255], always_apply=True),
    A.RandomCrop(height=IMG_HEIGHT, width=IMG_WIDTH, always_apply=True),
    A.Cutout(num_holes=1, max_h_size=112, max_w_size=112, fill_value=128, always_apply=True),
]

train_transform = transforms.Compose([
    np.uint8,
    transforms.Lambda(lambda x: np.array([x, x, x]).transpose((1, 2, 0)) ),
    np.uint8,
    Albumentations(preprocess + augmentations),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
#     transforms.ToPILImage(),
])
valid_transform = transforms.Compose([
    np.uint8,
    transforms.Lambda(lambda x: np.array([x, x, x]).transpose((1, 2, 0)) ),
    np.uint8,
    Albumentations(preprocess),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
#     transforms.ToPILImage(),
])

SVHN_transform = transforms.Compose([
    transforms.Lambda(lambda x: np.array([x, x, x]).transpose((1, 2, 0)) ),
    # convert image from grey-scale to RGB 3 channels
    np.uint8,
    transforms.ToPILImage(),
    transforms.AutoAugment(AutoAugmentPolicy.SVHN),
    transforms.ToTensor()])

no_transform = transforms.Compose([
    transforms.Lambda(lambda x: np.array([x, x, x]).transpose((1, 2, 0)) ),
    # convert image from grey-scale to RGB 3 channels
    np.uint8,
    transforms.ToPILImage(),
    transforms.ToTensor()])

class GraphemeDataset(torch.utils.data.Dataset):

    def __init__(self, info, images, transform=None,
                 num_grapheme_root=168, num_vowel_diacritic=11, num_consonant_diacritic=8):
        self.info = info
        self.grapheme_root_list = np.array(info['grapheme_root'].tolist(), dtype=np.int64)
        self.vowel_diacritic_list = np.array(info['vowel_diacritic'].tolist(), dtype=np.int64)
        self.consonant_diacritic_list = np.array(info['consonant_diacritic'].tolist(), dtype=np.int64)
        self.num_grapheme_root = num_grapheme_root
        self.num_vowel_diacritic = num_vowel_diacritic
        self.num_consonant_diacritic = num_consonant_diacritic
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        grapheme_root = self.grapheme_root_list[idx]
        vowel_diacritic = self.vowel_diacritic_list[idx]
        consonant_diacritic = self.consonant_diacritic_list[idx]
        label = (grapheme_root * self.num_vowel_diacritic + vowel_diacritic) * self.num_consonant_diacritic + consonant_diacritic
        np_image = self.images[idx].copy()
        out_image = self.transform(np_image)
        return out_image, label