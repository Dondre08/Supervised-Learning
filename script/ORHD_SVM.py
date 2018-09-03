# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC

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

# Model Complexity experiments--kernel rbf with different gamma
title = "ORHD: Model Complexity (SVM)"
gamma = np.logspace(-6, 1.2, 20)

train_scores, test_scores = validation_curve(SVC(random_state=random_state), 
                                             train_X, train_y, 
                                             param_name="gamma",
                                             param_range=gamma, cv=cv)

train_scores = 1 - train_scores
test_scores = 1 - test_scores


plt.figure()
plt.title(title)
plt.semilogx(gamma, train_scores.mean(axis=1), 'b', label="Training error", lw =2)
plt.semilogx(gamma, test_scores.mean(axis=1), 'g', label="Cross-validation error", lw =2)
plt.ylabel('Error')
plt.xlabel('Gamma of kernel')
plt.legend(loc="best")
plt.grid()

# fit the model by SVC (rbf)
gamma = 0.0025 # based on above experiment
classifier = SVC(gamma = gamma, random_state=random_state)
start_time = time.time()
classifier.fit(train_X, train_y)
prediction = classifier.predict(test_X) #Predict with test set
total_time = time.time() - start_time
print("Accuracy on training set (rbf): "+ str(classifier.score(train_X, train_y))) # score on train set
print("Run time (rbf): " + str(total_time) + " sec") # Run time




# Model Complexity experiments--kernel poly with different degree
title = "ORHD: Model Complexity (SVM)"
kernel = 'poly'
degree = np.arange(3,10,1)

train_scores, test_scores = validation_curve(SVC(random_state=random_state,
                                                 kernel = kernel), 
                                             train_X, train_y, 
                                             param_name="degree",
                                             param_range=degree, cv=cv)

train_scores = 1 - train_scores
test_scores = 1 - test_scores

plt.figure()
plt.title(title)
plt.plot(degree, train_scores.mean(axis=1), 'b', label="Training error")
plt.plot(degree, test_scores.mean(axis=1), 'g', label="Cross-validation error")
plt.ylabel('Error')
plt.xlabel('Degree of kernel')
plt.legend(loc="best")
plt.grid()

# fit the model by SVC (poly)
classifier = SVC(kernel = 'poly', degree = 4, random_state=random_state)
start_time = time.time()
classifier.fit(train_X, train_y)
prediction = classifier.predict(test_X) #Predict with test set
total_time = time.time() - start_time
print("Accuracy on training set (poly): "+ str(classifier.score(train_X, train_y))) # score on train set
print("Run time (poly): " + str(total_time) + " sec") # Run time

# fit the model by SVC (linear)
classifier = SVC(kernel = 'linear', random_state=random_state)
start_time = time.time()
classifier.fit(train_X, train_y)
prediction = classifier.predict(test_X) #Predict with test set
total_time = time.time() - start_time
print("Accuracy on training set (linear): "+ str(classifier.score(train_X, train_y))) # score on train set
print("Run time (linear): " + str(total_time) + " sec") # Run time



# Learning curve experiments (compare three different kernels)
title = "ORHD: Learning Curves (SVM)"
n_jobs = -1
train_sizes = np.linspace(.1, 1.0, 10)
gamma = 0.0025 # based on above experiment
degree = 4 # based on above experiment

plt.figure()
plt.title(title)
plt.xlabel("Training examples")
plt.ylabel("Error")

train_sizes, train_scores, test_scores = learning_curve(
                SVC(gamma = gamma, random_state=random_state), 
                train_X, train_y, cv=cv, n_jobs=n_jobs, 
                train_sizes=train_sizes, random_state=random_state)
            
train_scores = 1 - train_scores
test_scores = 1 - test_scores
            
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', 
                     label="Training error (rbf)", color = "r")
plt.plot(train_sizes, test_scores_mean, 'o-', 
                     label="Cross-validation error (rbf)", color = "r", ls= "dotted")

train_sizes, train_scores, test_scores = learning_curve(
                SVC(kernel = 'poly', degree = degree, random_state=random_state), 
                train_X, train_y, cv=cv, n_jobs=n_jobs, 
                train_sizes=train_sizes, random_state=random_state)
            
train_scores = 1 - train_scores
test_scores = 1 - test_scores
            
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', 
                     label="Training error (poly)", color = "g")
plt.plot(train_sizes, test_scores_mean, 'o-', 
                     label="Cross-validation error (poly)", color = "g", ls= "dotted")


train_sizes, train_scores, test_scores = learning_curve(
                SVC(kernel = 'linear', random_state=random_state), 
                train_X, train_y, cv=cv, n_jobs=n_jobs, 
                train_sizes=train_sizes, random_state=random_state)
            
train_scores = 1 - train_scores
test_scores = 1 - test_scores
            
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', 
                     label="Training error (linear)", color = "b")
plt.plot(train_sizes, test_scores_mean, 'o-', 
                     label="Cross-validation error (linear)", color = "b", ls= "dotted")

plt.legend(loc="best")
plt.grid()










# fit the finaly model by SVC (poly and degree =4)
classifier = SVC(kernel = 'poly', degree=4, random_state=random_state)
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
