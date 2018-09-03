# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.ensemble import AdaBoostClassifier

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

# Model Complexity experiments--different estimators
title = "ORHD: Model Complexity (Boosting)"
n_estimators = np.arange(1, 100, 2)
train_scores, test_scores = validation_curve(AdaBoostClassifier(random_state=random_state), 
                                             train_X, train_y, 
                                             param_name="n_estimators",
                                             param_range=n_estimators, cv=cv)

train_scores = 1 - train_scores
test_scores = 1 - test_scores


plt.figure()
plt.title(title)
plt.plot(n_estimators, train_scores.mean(axis=1), 'b', label="Training error")
plt.plot(n_estimators, test_scores.mean(axis=1), 'g', label="Cross-validation error")
plt.ylabel('Error')
plt.xlabel('Number of estimators')
plt.legend(loc="best")
plt.grid()

# Model Complexity experiments--different algorithm
n_estimators = 20 # based on the above experiment
classifierA = AdaBoostClassifier(n_estimators = n_estimators, random_state=random_state
                                ,algorithm = "SAMME.R")
classifierA.fit(train_X, train_y)
print("Accuracy on training set(SAMME.R): "+ str(classifierA.score(train_X, train_y))) # score on train set

classifierB = AdaBoostClassifier(n_estimators = n_estimators, random_state=random_state
                                ,algorithm = "SAMME")
classifierB.fit(train_X, train_y)
print("Accuracy on training set(SAMME): "+ str(classifierB.score(train_X, train_y))) # score on train set
classifierA.score(train_X, train_y)-classifierB.score(train_X, train_y)

# Model Complexity experiments--different learning rate & n_estimators were tested
algorithm = "SAMME" #based on above experiment result
classifierA = AdaBoostClassifier(n_estimators = 60, random_state=random_state
                                ,algorithm = algorithm, learning_rate = 0.5)
classifierA.fit(train_X, train_y)
print("Accuracy on training set(SAMME.R): "+ str(classifierA.score(train_X, train_y))) # score on train set



# Learning curve experiments
title = "ORHD: Learning Curves (Boosting)"
n_jobs = -1
train_sizes = np.linspace(.1, 1.0, 10)
n_estimators = 60 # basaed on previous experiment
learning_rate = 0.6 # based on previous experiment

plt.figure()
plt.title(title)
plt.xlabel("Training examples")
plt.ylabel("Error")

train_sizes, train_scores, test_scores = learning_curve(
                AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state,
                                   algorithm = algorithm, learning_rate = learning_rate), 
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


# fit the model by Adaboost
classifier = AdaBoostClassifier(n_estimators = n_estimators, random_state=random_state
                                ,algorithm = algorithm, learning_rate = learning_rate)
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
