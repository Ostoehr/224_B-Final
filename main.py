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
import matplotlib.pyplot as plt
from tensorflow import keras

BACKBONE = 'resnet18' # try a couple of different backbones and see what works the best
preprocess_input = sm.get_preprocessing(BACKBONE)



train_images = []
for directory_path in glob.glob("trainImages/trainImages"):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path)
        train_images.append(img)

train_images = np.array(train_images)


train_masks = []
for directory_path in glob.glob("trainMasks/trainMasks"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path, 0)
        train_masks.append(mask)

train_masks = np.array(train_masks)

X = train_images
Y = train_masks


# define number of channels
N = X.shape[-1]

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)

base_model = sm.Unet(BACKBONE, encoder_weights='imagenet')                      # probably can augment this somehow. choose loss function and optimizer
inp = keras.layers.Input(shape=(None, None, N))

l1 = keras.layers.Conv2D(3, (1, 1), padding="same", activation='relu')(inp) # map N channels data to 3 channels
out = base_model(l1)
out = MaxPooling2D((1,1))(out)
model = keras.models.Model(inp, out, name=base_model.name)


model.compile('Adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])

# tf.keras.utils.plot_model(model)

print(model.summary())

#checkpoint = keras.callbacks.ModelCheckpoint("model_1060_1_c1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
#early = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
#callbacks = [checkpoint, early]

history = model.fit(X_train,
                    Y_train,
                    batch_size=15,
                    epochs=6,
                    verbose=1,
                    validation_data=(X_val, Y_val))

model.save('test.h5')

# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
