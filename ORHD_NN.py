# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def read_data(data_file):
    data = pd.read_table(data_file, sep=',')

    attributes = len(data.values[0])

    X = data.values[0:, 0:attributes-1]
    y = data.values[0:, attributes-1]

    return X, y

# Load data
train_X, train_y = read_data('data\optdigits.tra')
test_X, test_y = read_data('data\optdigits.tes')
random_state=123

cv= 5

#Scale data 
scaler = StandardScaler()  
scaler.fit(train_X)  
train_X = scaler.transform(train_X)  
test_X = scaler.transform(test_X)  # apply same transformation to test data

# Model Complexity experiments--hidden layer size
title = "ORHD: Model Complexity (Neural Network)"
hidden_layer_sizes = np.arange(1, 20, 1) # with one hidden layer only

train_scores, test_scores = validation_curve(MLPClassifier(random_state=random_state, 
                                                           solver="lbfgs"), 
                                             train_X, train_y, 
                                             param_name="hidden_layer_sizes",
                                             param_range=hidden_layer_sizes, cv=cv)

train_scores = 1 - train_scores
test_scores = 1 - test_scores


plt.figure()
plt.title(title)
plt.plot(hidden_layer_sizes, train_scores.mean(axis=1), 'b', label="Training error")
plt.plot(hidden_layer_sizes, test_scores.mean(axis=1), 'g', label="Cross-validation error")
plt.ylabel('Error')
plt.xlabel('Hidden layer size')
plt.legend(loc="best")
plt.grid()

# Model complexity experiment-- test different solver sgd/ adam
# MLPClassifier(random_state=random_state, solver="sgd") 
# MLPClassifier(random_state=random_state, solver="adam") 
# Both algorithm are optimizing by a stepwise convergence to a minimum and 
# minimum wasn't found in max 200 iterations.
# So I've greyed out this part. 


# Learning curve experiments
title = "ORHD: Learning Curves (Nerual Network)"
hidden_layer_sizes = 10 # based on above experiment
solver="lbfgs"
n_jobs = -1
train_sizes = np.linspace(.1, 1.0, 10)

plt.figure()
plt.title(title)
plt.xlabel("Training examples")
plt.ylabel("Error")

train_sizes, train_scores, test_scores = learning_curve(
                MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, solver = solver, random_state=random_state), 
                train_X, train_y, cv=cv, n_jobs=n_jobs, 
                train_sizes=train_sizes, random_state=random_state)
            
train_scores = 1 - train_scores
test_scores = 1 - test_scores
            
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
    
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', 
                     label="Training error", color = "r")
plt.plot(train_sizes, test_scores_mean, 'o-', 
                     label="Cross-validation error", color = "g")

plt.legend(loc="best")
plt.grid()


# fit the model by MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, 
                           solver = solver,
                           random_state=random_state)
start_time = time.time()
classifier.fit(train_X, train_y)
prediction = classifier.predict(test_X) #Predict with test set
total_time = time.time() - start_time

# score on train set
print("Accuracy on training set: "+ str(classifier.score(train_X, train_y)))

#Accuracy, score on test set  
np.mean(prediction == test_y)
print("Accuracy on test set: " + str(classifier.score(test_X, test_y)))

# Run time
print("Run time: " + str(total_time) + " sec")

# Number of iterations
print("Total iterations: " + str(classifier.n_iter_) )