#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:13:21 2024

@author: alminagunduz
"""

from tensorflow.keras.utils import load_img,img_to_array
from tensorflow.keras.models import load_model
import numpy as np


from warnings import filterwarnings
from tensorflow import io
from tensorflow import image
from matplotlib import pyplot as plt



def image_loader(path):
    
    shape = [500,500]
    load = load_img(path, target_size = shape)
    input_arr = img_to_array(load)
    input_arr = np.array([input_arr])
    
    
    return input_arr

path1 = 'Data/test/crack/test_crack.jpg'
path2 = 'Data/test/missing-head/test_missinghead.jpg'
path3 = 'Data/test/paint-off/test_paintoff.jpg'

img1 = image_loader(path1)/255      
img2 = image_loader(path2)/255
img3 = image_loader(path3)/255


model = load_model("model_1.h5")

pred_img1= model.predict(img1)
pred_img2 = model.predict(img2)
pred_img3 = model.predict(img3)

filterwarnings("ignore") 
fig, ax = plt.subplots()


#FOR Crack
tf_img = io.read_file(path1)
tf_img = image.decode_png(tf_img, channels=3)
fig = plt.imshow(tf_img)
plt.title("True Crack Class: Crack")

# Define crack classes
crack_classes = {'crack': 0, 'missing-head': 1, 'paint-off': 2}

# Plot percentage text for each class
for label, index in crack_classes.items():
    value = pred_img1[0,index]*100
    ax.text(1150, 1450 + index*100, f"{label}: {value: .2f}%", c = 'green')


# Turn off axes for a cleaner display
ax.axis('off')

# Show the plot
plt.show()
   

#FOR Missing-Head 
fig, ax = plt.subplots()


tf_img = io.read_file(path2)
tf_img = image.decode_png(tf_img, channels=3)
fig = plt.imshow(tf_img)
plt.title("True Crack Class: Missing-Head")

# Define crack classes
crack_classes = {'crack': 0, 'missing-head': 1, 'paint-off': 2}

# Plot percentage text for each class
for label, index in crack_classes.items():
    value = pred_img2[0,index]*100
    ax.text(1150, 1450 + index*100, f"{label}: {value: .2f}%", c = 'green')


# Turn off axes for a cleaner display
ax.axis('off')

# Show the plot
plt.show()

#FOR Paint-off
fig, ax = plt.subplots()


tf_img = io.read_file(path2)
tf_img = image.decode_png(tf_img, channels=3)
fig = plt.imshow(tf_img)
plt.title("True Crack Class: Paint-off")

# Define crack classes
crack_classes = {'crack': 0, 'missing-head': 1, 'paint-off': 2}

# Plot percentage text for each class
for label, index in crack_classes.items():
    value = pred_img3[0,index]*100
    ax.text(1150, 1450 + index*100, f"{label}: {value: .2f}%", c = 'green')


# Turn off axes for a cleaner display
ax.axis('off')

# Show the plot
plt.show()