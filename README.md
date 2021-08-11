# Mailtruck-Classification
Image Classification for USPS and Amazon Vehicles using transfer-learning.\
Additionally, this is a good straight-forward example of how to get started with Image Classifcation and CNNs.

# Requirements
Only requirements include NumPy and Keras. You can install both packages with pip. 

# Features
This Classification model uses Imagenet's weights, a well known trained model in order to classify Amazon vans, USPS mailtrucks and cars.\
The dataset includes 108 images of cars and Amazon vans and 110 images of USPS mailtrucks.

# How to use:
The classifier automatically uses image augementation when loading images into the dataset.\
Make sure the data folder is in the same location as classifier.py or you must specify a new path to load the images in from.\
Once done, the model will train on 10 epochs. The classifier also includes a basic peice of code to load a single image and classify it.
If the program outputs 0, it predicted an amazon van, if it outputted 1, it predicted a car and if it outputted 2 it predicted a USPS mailtruck.
