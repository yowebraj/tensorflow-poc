import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#this dataset has each image mapped to a single label
data = keras.datasets.fashion_mnist
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

#we will be able to return data in this format:
(train_images, train_labels), (test_images, test_labels) = data.load_data()

#sequence of layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")        #picks values for each neurons
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#how many times this model will see the same image in a different order
#this will be an attempt to increase the accuracy
model.fit(train_images, train_labels, epochs=5)

prediction = model.predict(test_images)

#5 random #s from 1-10000
rand_num_list = np.random.randint(10000, size=5)

#choose 5 random clothes from dataset to analyze
for i in rand_num_list:
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction" + class_names[np.argmax(prediction[i])])
    plt.show()