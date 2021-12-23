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


os.chdir('/student/hlodhi1/Desktop/Face_Mask_Detection/capstone3/src/')

# # instantiate image data generators
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

# instantiate base model (mobilenetV2)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))


# define head layers for transfer learning
def add_new_last_layer(base_model, nb_classes=2):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    # Get the output shape of the models last layer
    x = base_model.output
    # construct new head for the transferred model
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten(name="flatten")(x) 
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# This will freeze the weights on all the layers except for our new dense layer
def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=1e-4, decay=1e-4 / 1000),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# set up transfer learning
train_model = add_new_last_layer(base_model)
setup_to_transfer_learn(train_model,base_model)

# Set up Tensorboard for logging models
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir(root_logdir)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

# Set up checkpoints 
import time
timestamp = time.strftime('%Y_%m_%d-%H_%M_%S')
checkpoint_filepath = '/other_models_w_scripts/mobilenetv2_mod_adam:(' + timestamp + '):.h5'
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)

# train the model
train_model.fit(
    train_generator,
    epochs= 100,
    validation_data= validation_generator,
    callbacks = [checkpoint_cb,early_stopping_cb,tensorboard_cb])