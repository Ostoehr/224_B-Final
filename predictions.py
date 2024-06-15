
from random import random
import numpy as np
import os as os
import tensorflow as tf
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import cv2
import glob
from tensorflow import keras
import matplotlib.pyplot as plt

# Define a function to display images
def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()

# Define backbone for U-net and get preprocessing function
BACKBONE = 'resnet18'                                                          
preprocess_input = sm.get_preprocessing(BACKBONE)
# Put test images into an array (correct relative path if needed)
test_images = []
image_paths = []
for directory_path in glob.glob("testImages/testImages"):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path)
        test_images.append(img)
        image_paths.append(img_path)

test_images = np.array(test_images)
# Preprocess the test images
test_images = preprocess_input(test_images)
# Load the trained model
model = keras.models.load_model('test.h5', compile=False)

# Change the current working directory to a directory to store the predicted masks
os.chdir(r"C:\Users\ostoehr\Desktop\Programming\Python\224_B-Final\testMasks")
X = 0
# Make predictions on the test images
predicted = model.predict(test_images)

# images = predicted.numpy()
print(len(predicted))
display([test_images[0], predicted[0]])

# Ensure the data type of the array is uint8
images = (predicted * 255).astype(np.uint8)
# Iterate over the images in the batch
for i, image in enumerate(predicted):
    normal = test_images[i]
    thresholder_image = np.where(image >= .5, 255, 0)
    if random() < .07:
        display([normal, thresholder_image])
    # Write each image to a file
    cv2.imwrite(f'{os.path.basename(image_paths[i]).replace(".jpg", "")}_output_image_.png', thresholder_image)

