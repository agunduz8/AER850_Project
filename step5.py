#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:13:21 2024

@author: alminagunduz
"""

from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow import io, image
from matplotlib import pyplot as plt
from warnings import filterwarnings

# Suppress warnings
filterwarnings("ignore")

# Load and preprocess the image
def image_loader(path, shape=(500, 500)):
    load = load_img(path, target_size=shape)
    input_arr = img_to_array(load) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(input_arr, axis=0)

# Paths to test images
image_paths = {
    "Crack": "Data/test/crack/test_crack.jpg",
    "Missing-Head": "Data/test/missing-head/test_missinghead.jpg",
    "Paint-Off": "Data/test/paint-off/test_paintoff.jpg"
}

# Load model
model = load_model("model_1.h5")

# Define crack classes
crack_classes = {'Crack': 0, 'Missing-head': 1, 'Paint-off': 2}

# Loop through each test image
for true_class, img_path in image_paths.items():
    
    # Load and preprocess the image
    img = image_loader(img_path)
    
    # Predict using the model
    prediction = model.predict(img)
    
    # Load the image for display
    tf_img = io.read_file(img_path)
    tf_img = image.decode_png(tf_img, channels=3)
    
    # Plot the image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(tf_img)
    ax.set_title(f"True Class: {true_class}", fontsize=16)
    
    # Display prediction probabilities
    for label, index in crack_classes.items():
        value = prediction[0, index] * 100
        ax.text(10, 30 + index * 30,f"{label}: {value:.2f}%", c='red',fontsize=14)
    
    # Turn off axes
    ax.axis('off')
    
    # Show the plot
    plt.show()