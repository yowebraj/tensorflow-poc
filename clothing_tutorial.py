import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#this dataset has each image mapped to a single label
#https://www.tensorflow.org/tutorials/keras/classification
data = keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

#we will be able to return data in this format:
(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images/255.0
#train_labels[7] is a pullover (label = 2)
# print(train_labels[7])
# print(train_images[7])

#infrared(?) aka default
#plt.imshow(train_images[7])
#black and white:
#plt.imshow(train_images[7], cmap=plt.cm.binary)
#plt.show()