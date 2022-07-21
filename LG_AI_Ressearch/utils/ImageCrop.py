import os
import  cv2
import json
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm
from glob import glob



def parser():
    parser = argparse.ArgumentParser(description="Using Crop Images")
    parser.add_argument('--csv', type=str, help='Glob train csv file')
    parser.add_argument('--image', type=str, help='Glob train image file')
    parser.add_argument('--json', type=str, help='Glob train json file')

    return parser.parse_args

def crop_image(bboxes, csv_file, show_image=False):
  print('===== Cropping images =====')
  for i in tqdm(range(len(bboxes))):
    image = cv2.imread(csv_file[i])
    h, w, x, y = int(bboxes[i]['h']), int(bboxes[i]['w']), int(bboxes[i]['x']), int(bboxes[i]['y'])
    crop_image = image[int(y):int(y+h), int(x):int(x+w)]
    if show_image:
        cv2.imshow('res', crop_image)
    cv2.imwrite(csv_file[i], crop_image)


if __name__ == '__main__':

    args = parser()
    
    train_csv = sorted(glob(args.csv))
    train_jpg = sorted(glob(args.image))
    train_json = sorted(glob(args.json))
    

    target_bboxes = []
    disease_bboxes = []

    for i in tqdm(range(len(train_json)) ,desc='Cropping Images'):
      with open(train_json[i], 'r') as f:
        data = json.load(f)
        target_bbox = data['annotations']['bbox'][0]
        disease_bbox = data['annotations']['part']
        target_bboxes.append(target_bbox)
        disease_bboxes.append(disease_bbox)
        
    crop_image(target_bboxes, train_jpg)
