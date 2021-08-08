import os
import tensorflow as tf
from tensorflow.keras.utils import normalize
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=.2,
        zoom_range=.2,
        horizontal_flip=True)
 
test_datagen = ImageDataGenerator(rescale=1./255)
 
training_set = test_datagen.flow_from_directory(
    'data/mailtruck_images/training_set',
    target_size=(256,256),
    batch_size = 32,
    class_mode = 'categorical')
 
test_set = test_datagen.flow_from_directory(
    'data/mailtruck_images/test_set',
    target_size=(256,256),
    batch_size = 32,
    class_mode = 'categorical')
 
base_model = keras.applications.VGG16(weights='imagenet',include_top=False,input_shape=(256,256,3))
for layer in base_model.layers[:-1]:
    layer.trainable = False
 
new_model  = keras.Sequential()
new_model.add(base_model)
new_model.add(Flatten())
new_model.add(Dense(units = 1024,activation='relu'))
new_model.add(Dropout(0.5))
new_model.add(Dense(units = 3,activation='softmax'))
 
new_model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
 
new_model.fit(
        training_set,
        steps_per_epoch=5,
        epochs=10,
        validation_data=test_set,
        validation_steps=800)
 
test_image = image.load_img('mailtruckvan.png',target_size=(256,256))
test_image = image.img_to_array(test_image)
test_image /= 255.0
test_image = np.expand_dims(test_image, axis = 0)
result = new_model.predict(test_image)
print(np.argmax(result))
