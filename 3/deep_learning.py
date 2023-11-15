#------------------------------------------------------------*/
# AUTHOR: Cameron Ross
# FILENAME: deep_learning.py
# SPECIFICATION: Question 3
# FOR: CS 4210- Assignment #4
# TIME SPENT: 30 min
#-----------------------------------------------------------*/

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def build_model(n_hidden, n_neurons_hidden, n_neurons_output, learning_rate):
    # Creating the Neural Network using the Sequential API
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))   #input layer
    # iterate over the number of hidden layers to create the hidden layers:
    for i in range(n_hidden):
        # hidden layer with ReLU activation function
        model.add(keras.layers.Dense(n_neurons_hidden, activation="relu")) 
    # output layer with one neural for each class and the softmax activation function since the classes are exclusive
    model.add(keras.layers.Dense(n_neurons_output, activation="softmax")) 
    # defining the learning rate
    opt = keras.optimizers.SGD(learning_rate)
    #Compiling the Model specifying the loss function and the optimizer to use.
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


# Using Keras to Load the Dataset. Every image is represented as a 28×28 array rather than a 1D array of size 784. 
# Moreover, the pixel intensities are represented as integers (from 0 to 255) rather than floats (from 0.0 to 255.0).
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
# creating a validation set and scaling the features
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
# For Fashion MNIST, we need the list of class names to know what we are dealing with. For instance, class_names[y_train[0]] = 'Coat'
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# Iterate over number of hidden layers, number of neurons in each hidden layer and the learning rate.
n_hidden = [2, 5, 10]
n_neurons = [10, 50, 100]
l_rate = [0.01, 0.05, 0.1]
highest_accuracy = 0
best_model = None
best_history = None
for i in range(len(n_hidden)):                          
    for j in range(len(n_neurons)):                      
        for k in range(len(l_rate)):                  
            # build the model for each combination
            model = build_model(n_hidden[i], n_neurons[j], len(class_names), l_rate[k])
            # To train the model
            # epochs = number times that the learning algorithm will work through the entire training dataset.
            history = model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))
            # Calculate the accuracy of this neural network and store its value if it is the highest so far. To make a prediction, do:
            predictions = np.argmax(model.predict(X_test), axis=-1)
            correct_preds = 0
            for p in range(len(predictions)):
                if predictions[p] == y_test[p]: 
                    correct_preds = correct_preds + 1
            current_accuracy = correct_preds / len(predictions)
            print("**************")
            print(predictions)
            print(y_test)
            print("**************")
            #-->add your Pyhton code here
            if current_accuracy >= highest_accuracy:
                highest_accuracy = current_accuracy
                best_model = model
                best_history = history        
                print(f"Highest accuracy so far: {highest_accuracy}")
                print(f"Parameters: number of Hidden Layers: {n_hidden[i]}, number of neurons: {n_neurons[j]}, learning rate: {l_rate[k]}")
            print()


# After generating all neural networks, print the summary of the best model found
# The model’s summary() method displays all the model’s layers, including each layer’s name (which is automatically generated unless you set it when creating the layer), its
# output shape (None means the batch size can be anything), and its number of parameters. Note that Dense layers often have a lot of parameters. This gives the model quite a lot of
# flexibility to fit the training data, but it also means that the model runs the risk of overfitting, especially when you do not have a lot of training data.

print(best_model.summary())
img_file = './model_arch.png'
tf.keras.utils.plot_model(model, to_file=img_file, show_shapes=True, show_layer_names=True)

# plotting the learning curves of the best model
pd.DataFrame(best_history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()
