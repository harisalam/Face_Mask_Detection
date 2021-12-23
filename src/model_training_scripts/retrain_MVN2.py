import os
import shutil

import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

os.chdir('/student/hlodhi1/Desktop/Face_Mask_Detection/capstone3/src/')

# instantiate image data generators
train_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
test_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
train_generator = train_datagen.flow_from_directory(
        '../data/dataset/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        '../data/dataset/val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle = False)
test_generator = test_datagen.flow_from_directory(
        '../data/dataset/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle = False)


# load model 
model = load_model("Models/MNV2_:(2021_12_18-16_24_57):.h5")

# Set up Tensorboard for logging models
root_logdir = os.path.join(os.curdir , "my_logs")
def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir(root_logdir)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

# Set up checkpoints 
import time
timestamp = time.strftime('%Y_%m_%d-%H_%M_%S')
checkpoint_filepath = 'Models/MNV2_:(' + timestamp + '):.h5'
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,save_best_only=True)
# changed patience from 10 to 15
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=15,restore_best_weights=True)


# re-train the model
model.fit(
    train_generator,
    epochs= 100,
    validation_data= validation_generator,
    callbacks = [checkpoint_cb,early_stopping_cb,tensorboard_cb])