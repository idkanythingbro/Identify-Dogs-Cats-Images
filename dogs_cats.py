import os
import zipfile
import random
import keras._tf_keras
import tensorflow as tf
import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.optimizers import RMSprop
from shutil import copyfile

#================================================================================================
# local_zip = "PetImages\kagglecatsanddogs_5340.zip"
# zip_ref   = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('/tmp')
# zip_ref.close()
# print(len(os.listdir('/tmp/PetImages/Cat/')))
# print(len(os.listdir('/tmp/PetImages/Dog/')))
#================================================================================================

#================================================================================================
# try:
#     os.mkdir('/tmp/cats-v-dogs')
#     os.mkdir('/tmp/cats-v-dogs/training')
#     os.mkdir('/tmp/cats-v-dogs/testing')
#     os.mkdir('/tmp/cats-v-dogs/training/cats')
#     os.mkdir('/tmp/cats-v-dogs/training/dogs')
#     os.mkdir('/tmp/cats-v-dogs/testing/cats')
#     os.mkdir('/tmp/cats-v-dogs/testing/dogs')
# except OSError:
#     pass

# def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
#     files = []
#     for filename in os.listdir(SOURCE):
#         file = SOURCE + filename
#         if os.path.getsize(file) > 0:
#             files.append(filename)
#         else:
#             print(filename + " is zero length, so ignoring.")
 
#     training_length = int(len(files) * SPLIT_SIZE)
#     testing_length = int(len(files) - training_length)
#     shuffled_set = random.sample(files, len(files))
#     training_set = shuffled_set[0:training_length]
#     testing_set = shuffled_set[:testing_length]
 
#     for filename in training_set:
#         this_file = SOURCE + filename
#         destination = TRAINING + filename
#         copyfile(this_file, destination)
 
#     for filename in testing_set:
#         this_file = SOURCE + filename
#         destination = TESTING + filename
#         copyfile(this_file, destination)
 
 
# CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
# TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
# TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
# DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
# TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
# TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"
 
# split_size = .9
# split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
# split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
#=========================================================================================================

# print(len(os.listdir('/tmp/cats-v-dogs/training/cats/')))
# print(len(os.listdir('/tmp/cats-v-dogs/training/dogs/')))
# print(len(os.listdir('/tmp/cats-v-dogs/testing/cats/')))
# print(len(os.listdir('/tmp/cats-v-dogs/testing/dogs/')))

#=========================================================================================================

model = keras._tf_keras.keras.models.Sequential([
        keras.layers.Conv2D(16, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
])
# model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# #============================================================================================================

# TRAINING_DIR = "/tmp/cats-v-dogs/training/"
# train_datagen = ImageDataGenerator(rescale=1.0/255.)
# train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
#                                                     batch_size=100,
#                                                     class_mode='binary',
#                                                     target_size=(150, 150))
 
# VALIDATION_DIR = "/tmp/cats-v-dogs/testing/"
# validation_datagen = ImageDataGenerator(rescale=1.0/255.)
# validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
#                                                               batch_size=100,
#                                                               class_mode='binary',
#                                                               target_size=(150, 150))
 



# history = model.fit(train_generator,
#                               epochs=15,
#                               verbose=1,
#                               validation_data=validation_generator)

# import matplotlib.image  as mpimg
# import matplotlib.pyplot as plt
# #-----------------------------------------------------------
# # Retrieve a list of list results on training and test data
# # sets for each training epoch
# #-----------------------------------------------------------
# acc=history.history['accuracy']
# val_acc=history.history['val_accuracy']
# loss=history.history['loss']
# val_loss=history.history['val_loss']
 
# epochs=range(len(acc)) # Get number of epochs
 
# #------------------------------------------------
# # Plot training and validation accuracy per epoch
# #------------------------------------------------
# plt.plot(epochs, acc, 'r', "Training Accuracy")
# plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
# plt.title('Training and validation accuracy')
# plt.figure()
 
# #------------------------------------------------
# # Plot training and validation loss per epoch
# #------------------------------------------------
# plt.plot(epochs, loss, 'r', "Training Loss")
# plt.plot(epochs, val_loss, 'b', "Validation Loss")
# plt.figure()

import numpy as np
import os
# from google.colab import files
import cv2
from keras._tf_keras.keras.preprocessing import image
 

 

 
  # predicting images
path = 'name.jpg'# move the image in the same path and enter the name 
img = image.load_img(path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
 
images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0]<0.5:
    print(" is a dog")
else:
    print(" is a cat")
