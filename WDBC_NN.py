# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


#load breast cancer data
breast = load_breast_cancer()
X, y = breast.data, breast.target
random_state=123

# Straitify the dataset (stratify = y), and split data set to 30% test, 70% train set
train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size=0.3,
                                                    random_state=random_state,
                                                    stratify=y)

cv= 5

#Scale data 
scaler = StandardScaler()  
scaler.fit(train_X)  
train_X = scaler.transform(train_X)  
test_X = scaler.transform(test_X)  # apply same transformation to test data


# Model Complexity experiments--hidden layer size
title = "WDBC: Model Complexity (Neural Network)"
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


# Learning curve experiments
title = "WDBC: Learning Curves (Nerual Network)"
hidden_layer_sizes = 5 # based on above experiment
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
