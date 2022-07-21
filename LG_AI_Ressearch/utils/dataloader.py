import json
import argparse

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2 

from glob import glob
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold


def parser():
    parser = argparse.ArgumentParser(description="Create Data Frame")
    parser.add_argument('--seed', type=int, default=42, help='Fix Seed')
    parser.add_argument('--train_csv', type=str, help='Glob train csv file')
    parser.add_argument('--train_image', type=str, help='Glob train image file')
    parser.add_argument('--train_json', type=str, help='Glob train json file')
    parser.add_argument('--test_image', type=str, help='Glob test image file')
    parser.add_argument('--foldnum', type=int, default=3, help='StratifiedKFold')
    parser.add_argument('--batch', type=int, default = 16, help='Batch Size')
    parser.add_argument('--worker', type=int, default = 1, help = 'torch.cuda.device_count() * 4')

    return parser.parse_args()

def create_dataframe():
    args = parser()

    train_csv = sorted(glob.glob(args.train_csv))
    train_jpg = sorted(glob(args.train_mage))
    train_json = sorted(glob(args.train_json))
    test_jpg = sorted(glob(args.test_image))

    crops = []
    diseases = []
    risks = []
    labels = []

    for i in tqdm(range(len(train_json)),desc='Read Json'):
        with open(train_json[i], 'r') as f:
            sample = json.load(f)
            crop = sample['annotations']['crop']
            disease = sample['annotations']['disease']
            risk = sample['annotations']['risk']
            label=f"{crop}_{disease}_{risk}"
        
            crops.append(crop)
            diseases.append(disease)
            risks.append(risk)
            labels.append(label)

    train_df = pd.DataFrame({'image_path': train_jpg,'label':labels})
    train_df['label_unique']= pd.factorize(train_df['label'])[0]
    test_df = pd.DataFrame({'image_path': test_jpg})

    return train_df, test_df 

def train_valid_dataframe():
  args = parser()
  train_df, test_df = create_dataframe()
  skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state = args.seed)
  for fold,(_,val) in enumerate(skf.split(X = train_df, y= train_df.label_unique)):
    train_df.loc[val, 'kfold'] = fold

    return train_df, test_df

def train_aug():
    tr_aug = A.Compose([
                        A.Resize(520, 520),
                        A.VerticalFlip(p=0.5),
                        A.HorizontalFlip(p=0.5),
                        A.ColorJitter(0.5),
                        A.ShiftScaleRotate(scale_limit=(0.7, 0.9), p=0.5, rotate_limit=30),
                        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
                        A.Blur(p=0.3),
                        A.Normalize(p=1),
                        ToTensorV2(p=1)
                        ])


    return tr_aug

def test_aug():
    ts_aug = A.Compose([
                        A.Resize(520, 520),
                        A.VerticalFlip(p=0.5),
                        A.HorizontalFlip(p=0.5),
                        A.Normalize(p=1),
                        ToTensorV2(p=1)
                        ])

    return ts_aug

class TrainDataset(Dataset):
  def __init__(self, df, augmentation = None):
    self.df = df
    self.augmentation = augmentation

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    img_path = self.df.image_path[index]
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    label = self.df.label_unique[index]

    if self.augmentation is not None:
      img = self.augmentation(image = img)['image']

    return img, label

class TestDataset(Dataset):
  def __init__(self, df, augmentation = None):
    self.df = df
    self.augmentation = augmentation

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    img_path = self.df.image_path[index]
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    if self.augmentation is not None:
      img = self.augmentation(image = img)['image']

    return img


def create_loader(df, fold_num):
  args = parser()
  train_df = df[df['kfold'] != fold_num].reset_index(drop=True)
  valid_df = df[df['kfold'] == fold_num].reset_index(drop=True)

  train_ds = TrainDataset(train_df, augmentation=train_aug)
  valid_ds = TrainDataset(valid_df, augmentation=train_aug)

  train_loader = DataLoader(train_ds, batch_size = args.batch, num_workers=args.worker, shuffle=True, pin_memory = True)
  valid_loader = DataLoader(valid_ds, batch_size = args.batch, num_workers=args.worker, shuffle=False, pin_memory = True)

  return train_loader, valid_loader
