## Import packages and libraries

from random import random
import numpy as np
import pandas as pd
import os as os
from classification_models.models.resnet import ResNet34
from keras import Model, Input
from keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, Flatten
os.environ["SM_FRAMEWORK"] = "tf.keras"
from PIL import Image
import tensorflow as tf
import segmentation_models as sm
import cv2
import glob
from sklearn.model_selection import train_test_split
from tensorflow import keras

# Define backbone for U-net and get preprocessing function

BACKBONE = 'resnet18' 
preprocess_input = sm.get_preprocessing(BACKBONE)

# Put training images into an array (correct relative path if needed)
train_images = []
for directory_path in glob.glob("trainImages/trainImages"):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path)
        train_images.append(img)

train_images = np.array(train_images)

# Put training masks into an array (correct relative path if needed)
train_masks = []
for directory_path in glob.glob("trainMasks/trainMasks"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path, 0)
        train_masks.append(mask)

train_masks = np.array(train_masks)

# Define X and Y
X = train_images
Y = train_masks


# Define number of channels
N = X.shape[-1]

# Split data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)

# Define model
base_model = sm.Unet(BACKBONE, encoder_weights='imagenet')
inp = keras.layers.Input(shape=(None, None, N))
l1 = keras.layers.Conv2D(3, (1, 1), padding="same", activation='relu')(inp) # map N channels data to 3 channels
out = base_model(l1)
out = MaxPooling2D((1,1))(out)
model = keras.models.Model(inp, out, name=base_model.name)

# Compile model and choose loss function
model.compile('Adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])

print(model.summary())

# Fit model
history = model.fit(X_train,
                    Y_train,
                    batch_size=15,
                    epochs=6,
                    verbose=1,
                    validation_data=(X_val, Y_val))
# Save model
model.save('test.h5')