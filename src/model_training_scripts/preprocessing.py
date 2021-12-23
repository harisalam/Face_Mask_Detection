import os
import shutil


import numpy as np
from PIL import Image
import tensorflow as tf
from imutils import paths
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


# Create directories for classes with training, testing, and validation
root_dir = '/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset'
Cls_mask = '/with_mask'
Cls_nomask = '/without_mask'
os.makedirs(root_dir +'/train' + Cls_mask)
os.makedirs(root_dir +'/train' + Cls_nomask)
os.makedirs(root_dir +'/val' + Cls_mask)
os.makedirs(root_dir +'/val' + Cls_nomask)
os.makedirs(root_dir +'/test' + Cls_mask)
os.makedirs(root_dir +'/test' + Cls_nomask)

# shuffle no mask data 
currentCls = Cls_nomask
src = "/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset"+currentCls
allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])
train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]
print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))

#shuffle mask data
currentCls = Cls_mask
src = "/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset"+currentCls
allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])
train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]
print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))

# copy files into respective directories
for name in train_FileNames:
    shutil.copy(name, "/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/train"+currentCls)
for name in val_FileNames:
    shutil.copy(name, "/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/val"+currentCls)
for name in test_FileNames:
    shutil.copy(name, "/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/test"+currentCls)


# rename files for convenience
for idx, filename in enumerate(os.listdir('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/test/with_mask')):
    os.rename('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/test/with_mask/' + f'{filename}', '/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/test/with_mask/' + 'with_mask' +f'{idx}.jpg')
for idx, filename in enumerate(os.listdir('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/test/without_mask')):
    os.rename('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/test/without_mask/' + f'{filename}', '/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/test/without_mask/' + 'without_mask' +f'{idx}.jpg')
for idx, filename in enumerate(os.listdir('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/train/with_mask')):
    os.rename('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/train/with_mask/' + f'{filename}', '/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/train/with_mask/' + 'with_mask' +f'{idx}.jpg')
for idx, filename in enumerate(os.listdir('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/train/without_mask')):
    os.rename('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/train/without_mask/' + f'{filename}', '/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/train/without_mask/' + 'without_mask' +f'{idx}.jpg')
for idx, filename in enumerate(os.listdir('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/val/with_mask')):
    os.rename('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/val/with_mask/' + f'{filename}', '/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/val/with_mask/' + 'with_mask' +f'{idx}.jpg')
for idx, filename in enumerate(os.listdir('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/val/without_mask')):
    os.rename('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/val/without_mask/' + f'{filename}', '/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/val/without_mask/' + 'without_mask' +f'{idx}.jpg')

# verify that no corrupt images exist and if so, remove them
for filename in os.listdir('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/train/with_mask'):
    try: 
        Image.open(f'/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/train/with_mask/{filename}')
    except: 
        os.remove(f'/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/train/with_mask/{filename}')
        print('removed')
for filename in os.listdir('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/train/without_mask'):
    try: 
        Image.open(f'/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/train/without_mask/{filename}')
    except: 
        os.remove(f'/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/train/without_mask/{filename}')
        print('removed')
for filename in os.listdir('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/test/with_mask'):
    try: 
        Image.open(f'/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/test/with_mask/{filename}')
    except: 
        os.remove(f'/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/test/with_mask/{filename}')
        print('removed')
for filename in os.listdir('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/test/without_mask'):
    try: 
        Image.open(f'/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/test/without_mask/{filename}')
    except: 
        os.remove(f'/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/test/without_mask/{filename}')
        print('removed')
for filename in os.listdir('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/val/with_mask'):
    try: 
        Image.open(f'/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/val/with_mask/{filename}')
    except: 
        os.remove(f'/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/val/with_mask/{filename}')
        print('removed')
for filename in os.listdir('/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/val/without_mask'):
    try: 
        Image.open(f'/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/val/without_mask/{filename}')
    except: 
        os.remove(f'/student/hlodhi1/Desktop/Face Mask Detection/capstone3/data/dataset/val/without_mask/{filename}')
        print('removed')