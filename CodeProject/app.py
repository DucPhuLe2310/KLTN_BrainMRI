import os
os.system('pip install tensorflow==2.9.1 tensorflow-addons==0.17.1')

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import tensorflow_addons as tfa

smooth = 1e-15

def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# Load the trained model
model_path = "D:\Data\KLTN_BrainMRI\Unet_improve\model.h5"
model = load_model(model_path, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef, 'GroupNormalization': tfa.layers.GroupNormalization})

# Function to preprocess image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict and post-process the mask
def predict(image):
    image = preprocess_image(image)
    mask = model.predict(image)[0]
    mask = (mask > 0.5).astype(np.uint8)
    mask = np.squeeze(mask, axis=-1)
    return mask

# Streamlit app
st.title("Brain MRI Segmentation with Attention U-Net")

uploaded_file = st.file_uploader("Choose a Brain MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    mask = predict(image)

    # Display the mask
    st.image(mask, caption='Predicted Mask.', use_column_width=True)
