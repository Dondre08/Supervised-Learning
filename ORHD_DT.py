# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

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

# Model Complexity experiments--different depth
title = "WDBC: Model Complexity (Decision Tree)"
max_depth = np.arange(1, 40, 1)
train_scores, test_scores = validation_curve(DecisionTreeClassifier(random_state=random_state), 
                                             train_X, train_y, 
                                             param_name="max_depth",
                                             param_range=max_depth, cv=cv)

train_scores = 1 - train_scores
test_scores = 1 - test_scores


plt.figure()
plt.title(title)
plt.plot(max_depth, train_scores.mean(axis=1), 'b', label="Training error")
plt.plot(max_depth, test_scores.mean(axis=1), 'g', label="Cross-validation error")
plt.ylabel('Error')
plt.xlabel('Max depth')
plt.legend(loc="best")
plt.grid()

max_depth = 10 # based on the above result
classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
classifier.fit(train_X, train_y)
GINI = classifier.score(train_X, train_y)
print("Accuracy on training set (GINI): "+ str(GINI))

# Model Complexity experiments--information gain
classifier = DecisionTreeClassifier(max_depth=max_depth, criterion="entropy", random_state=random_state)
classifier.fit(train_X, train_y)
information_gain = classifier.score(train_X, train_y)
print("Accuracy on training set (information gain): "+ str(information_gain))
print(GINI-information_gain)


# Learning curve experiments
title = "ORHD: Learning Curves (Decision Tree)"
n_jobs = -1
train_sizes = np.linspace(.1, 1.0, 10)

plt.figure()
plt.title(title)
plt.xlabel("Training examples")
plt.ylabel("Error")

train_sizes, train_scores, test_scores = learning_curve(
                DecisionTreeClassifier(max_depth = max_depth, criterion="entropy", random_state=random_state), 
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

# fit the model by DecisionTree
classifier = DecisionTreeClassifier(max_depth=max_depth, criterion="entropy", random_state=random_state)
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
