""" Prepares the dataset needed for deep learning models. Dataset is created from the images obtained by executing Vid2Img.py (Slices of 42 CT scan videos provided by JIPMER)"""
import os
import cv2
import pandas as pd
import numpy as np

## User Defined Inputs
# Directory of all images
IMG_DIR = os.getcwd() + '\\all_images'
# Image size(square)
IMG_SIZE = 224

# Create Patient ID-target mapping from given excel sheet
if os.path.exists('target.xlsx'):
        target = pd.read_excel('target.xlsx')
        print('Loaded target dataset')
else:
    # Read targets from excel file
    dataset = pd.read_excel('Head trauma scans training set 18 to 70 years.xlsx')
    # Only pick columns required for model i.e. ID of patient and target
    target = dataset[['Unique scan ID (Name,hosp number, time of scan, and decision) Pasted values on 28 april 2018', 'Operated y/n' ]]
    # Rename columns
    target = target.rename(columns={'Unique scan ID (Name,hosp number, time of scan, and decision) Pasted values on 28 april 2018': 'ID', 'Operated y/n': 'operated'})
    # Convert from bool to binary encoding
    target = target.dropna()
    target['operated'] = target['operated'].map({'y': 1, 'n': 0})
    target['operated'] = target['operated'].astype(int)
    target.to_excel('target.xlsx')
    print('Created target dataset')

# Assign label(target) to input image
def label_img(img):
    # Image file name is of the form 'AlaH-47767543194.2722222222n--2.jpg' i.e. ID--count.jpg
    img_ID = img.split('--')[0]
    temp = target[target['ID'] == img_ID]['operated'].tolist()
    label = temp + [i^1 for i in temp]
    return label

# Create Image/Slice - target dataset used for deep learning model. 
def create_img_dataset():
    X, y = [], []
    
    for img in os.listdir(IMG_DIR):
        # Assign label(target) to input image
        label = label_img(img)
        # if label==[]:
        #     print(img)
        # Read the image as grayscale and resize 
        img_path = os.path.join(IMG_DIR, img)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # Convert image into numpy array. Add [image, target] to dataset
        X.append(np.array(img))
        y.append(label)

    np.save('X.npy', X)
    np.save('y.npy', y)
    print('Created arrays X and y')
create_img_dataset()


