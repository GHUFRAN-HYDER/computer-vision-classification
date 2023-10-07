#!/usr/bin/env python
# coding: utf-8

# # Code Overview
# The following code loads image data and prepares it for a machine learning model that will be used to classify images as cats or dogs.
# 
# The code does the following:
# 
# Import necessary libraries for the code.
# 
# Define the size of the images and the batch size for training and testing.
# 
# Load the training and testing directories.
# 
# Create a data frame containing the file names and the corresponding category (cat or dog) for each image.
# 
# Split the data into training and validation sets.
# 
# Use the ImageDataGenerator to rescale the image data and create training, validation, and testing generators.
# 
# Plot sample images from the training set.
# 
# Define the model architecture.

# # Importing libraries
# The first step of the code is to import the required libraries.

# In[ ]:


import os
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Rescaling, Conv2D, MaxPool2D, Flatten, Dense
from PIL import Image
import os
import matplotlib.pyplot as plt 
plt.style.use("ggplot")
import seaborn as sns
from glob import glob


# In[ ]:





# In[2]:


# Define the image size and batch size
img_size = (150, 150)
batch_size = 32


# In[3]:


# Define the paths to the train and test directories
train_dir = 'D:/MLPROJECT/dogs-vs-cats/train/train'


# # Create a data frame 
# The code creates a data frame containing the file names and the corresponding category (cat or dog) for each image.

# In[5]:


file_names = glob('D:/MLPROJECT/dogs-vs-cats/train/train/*.jpg')
categories = [1 if 'dog' in pic else 0 for pic in os.listdir("D:/MLPROJECT/dogs-vs-cats/train/train")]

df = pd.DataFrame({'filename': file_names, 'category':categories})
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})

print("shape:", df.shape)
df.head()


# # Split the data 
# The code then splits the data into training and validation sets.

# In[6]:


from sklearn.model_selection import train_test_split

train_df, validate_df = train_test_split(df, test_size=0.2, random_state=10)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

# print shape
train_df.shape, validate_df.shape


# In[7]:


# plot label counts
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

sns.countplot(train_df.category, ax=ax[0])
ax[0].set_title('Training dataset', fontsize=14)

sns.countplot(validate_df.category, ax=ax[1])
ax[1].set_title('Valiadtion dataset', fontsize=14);


# In[8]:


file_names = os.listdir("D:/MLPROJECT/dogs-vs-cats/test1/test1")

test_df = pd.DataFrame({'filename': file_names})

print("shape:", test_df.shape)
test_df.head()


# In[9]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (200,200)
BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale = 1/255)


# In[10]:



train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
#     directory='./Images/train', 
    x_col='filename',
    y_col='category',
    class_mode='binary',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)


# In[11]:


valid_generator = datagen.flow_from_dataframe(
    dataframe=validate_df,
#     directory='./Images/train',
    x_col='filename',
    y_col='category',
    class_mode='binary',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)


# In[12]:


test_generator = datagen.flow_from_dataframe(
    dataframe=test_df,
    directory="D:/MLPROJECT/dogs-vs-cats/test1/test1",
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)


# In[13]:


# get a batch of 32 training images 
images = train_generator.next()[:9]

# plot 9 original training images
plt.figure(figsize=(5, 5))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[0][i])
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[14]:


# Design the model architecture
model = Sequential([
    Input(shape=(200,200,3)),
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPool2D(pool_size=2),
    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPool2D(pool_size=2),
    Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    MaxPool2D(pool_size=2),
    Conv2D(filters=265, kernel_size=3, padding='same', activation='relu'),
    Flatten(),
    Dense(1, activation='sigmoid')  # Cat or dog  
])

model.summary()


# In[15]:


# Compile model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[16]:


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# callbacks is a reguralization technique to brevent over fitting
callbacks = [
    # to stop training when you measure that the validation loss is no longer improving
    EarlyStopping(patience=4, monitor='val_loss'),
    # reduce learning_rate if the model is not imporving
    ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001),
    # save best model
    ModelCheckpoint(filepath='D:/MLPROJECT/dogs-vs-cats/Models/model.keras', save_best_only=True, monitor='val_loss')
]

# train the model
history = model.fit(train_generator, validation_data=valid_generator, epochs=10, callbacks=[callbacks])


# In[17]:


# plot model performance

pd.DataFrame(history.history).plot();


# In[19]:


from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import load_img
import tensorflow as tf
#Input image
test_image = tf.keras.utils.load_img ('D:/MLPROJECT/dogs-vs-cats/test1/test1/1.jpg',target_size=(200,200))
  
#For show image
plt.imshow(test_image)
test_image = tf.keras.utils.img_to_array(test_image)
             
test_image = np.expand_dims(test_image,axis=0)
  
# Result array
result = model.predict(test_image)
  
#Mapping result array with the main name list
i=0
if(result>=0.5):
  print("Dog")
else:
  print("Cat")


# In[ ]:


import tensorflow as tf
from pyspark.sql import SparkSession

spark = (SparkSession.builder.getOrCreate())
model =tf.keras.models.loadmodel('D:\MLPROJECT\dogs-vs-cats\Models', compile=False)

