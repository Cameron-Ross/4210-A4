#-------------------------------------------------------------------------
# AUTHOR: Cameron Ross
# FILENAME: deep_learning.py
# SPECIFICATION: Question 2
# FOR: CS 4210- Assignment #4
# TIME SPENT: 30 min
#-----------------------------------------------------------*/

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]
algos=['single', 'multi']
# reading the training data by using Pandas library
training_data = pd.read_csv('optdigits.tra', sep=',', header=None) 
# getting the first 64 fields to form the feature data for training
X_training = np.array(training_data.values)[:,:64] 
# getting the last field to form the class label for training
y_training = np.array(training_data.values)[:,-1]  
# reading the test data by using Pandas library
test_data = pd.read_csv('optdigits.tes', sep=',', header=None) 
# getting the first 64 fields to form the feature data for test
X_test = np.array(test_data.values)[:,:64] 
# getting the last field to form the class label for test   
y_test = np.array(test_data.values)[:,-1]  


highest_single_accuracy = 0
highest_multi_accuracy = 0
best_single_lr = None
best_multi_lr = None
best_single_shuf = None
best_multi_shuf = None

for i in range(len(n)):
    for j in range(len(r)):
        for k in range(len(algos)):
            algo = algos[k]
            if algo == 'single':
                clf = Perceptron(eta0=n[i], shuffle=r[j], max_iter=1000)    
            else:
                # hidden_layer_sizes = number of neurons in the ith hidden layer,
                clf = MLPClassifier(activation='logistic', max_iter=1000, learning_rate_init=n[i], shuffle=r[j])                           
            #Fit the Neural Network to the training data
            clf.fit(X_training, y_training)
            # Make the prediction for each test sample
            y_pred = clf.predict(X_test)
            # Compute accuracy
            correct_preds = 0
            for p in range(len(y_pred)):
                if y_pred[p] == y_test[p]:
                    correct_preds = correct_preds + 1
                current_accuracy = correct_preds / len(y_pred)
                # check if the calculated accuracy is higher than the previously one calculated. 
                if (current_accuracy > highest_single_accuracy and algo == 'single') or (current_accuracy > highest_multi_accuracy and algo == 'multi'):
                    # If so, update the highest accuracy
                    if algo == 'single': 
                        highest_single_accuracy = current_accuracy
                        best_single_shuf = r[j]
                        best_single_lr = n[i]
                    else: 
                        highest_multi_accuracy = current_accuracy
                        best_multi_shuf = r[j]
                        best_multi_lr = n[i]
                    # Print
                    print('-----------------------------------------------------------------------------------------------')
                    if (best_single_lr != None): print(f"Highest SLP accuracy so far: {highest_single_accuracy}, Parameters: learning rate={best_single_lr}, shuffle={best_single_shuf}")
                    if (best_multi_lr != None): print(f"Highest MLP accuracy so far: {highest_multi_accuracy}, Parameters: learning rate={best_multi_lr}, shuffle={best_multi_shuf}")
                    print('-----------------------------------------------------------------------------------------------')



# Questions
    # Print the learning rate and shuffle of each algo (so need to keep track seperately)
    # Calculate both accuracies seperately correct??
    # hidden_layer_sizes - how do you calculate
    # is r for shuffle?
