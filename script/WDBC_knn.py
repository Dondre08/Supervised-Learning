# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

# load breast cancer data
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

# Model Complexity experiments- test with different k
title = "WDBC: Model Complexity (KNN)"
n_neighbors = [1, 3, 9, 20, 50]
train_scores, test_scores = validation_curve(KNeighborsClassifier(), 
                                             train_X, train_y, 
                                             param_name="n_neighbors",
                                             param_range=n_neighbors, cv=cv)

train_scores = 1 - train_scores
test_scores = 1 - test_scores


plt.figure()
plt.title(title)
plt.plot(n_neighbors, train_scores.mean(axis=1), 'b', label="Training error")
plt.plot(n_neighbors, test_scores.mean(axis=1), 'g', label="Cross-validation error")
plt.ylabel('Error')
plt.xlabel('Number of neighbors')
plt.xlim([50, 0])#Note that many neighbors mean a "smooth" or "simple" model, so the plot uses a reverted x axis.
plt.legend(loc="best")
plt.grid()

# fit the model by KNN (k=3 based on the rsult of above learing curve)
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(train_X, train_y)
print("Accuracy on training set: "+ str(classifier.score(train_X, train_y))) # score on train set

# Model Complexity experiments- test with different weights
classifier = KNeighborsClassifier(n_neighbors=3, weights='distance')
classifier.fit(train_X, train_y)
print("Accuracy on training set: "+ str(classifier.score(train_X, train_y))) # score on train set




# Learning curve experiments
title = "WDBC: Learning Curves (KNN)"
n_jobs = -1
train_sizes = np.linspace(.1, 1.0, 10)
p=2

plt.figure()
plt.title(title)
plt.xlabel("Training examples")
plt.ylabel("Error")

for k in (1, 3,9,20): # k with different values
        train_sizes, train_scores, test_scores = learning_curve(
                KNeighborsClassifier(n_neighbors=k, p=p, weights='distance'), train_X, train_y, cv=cv, n_jobs=n_jobs, 
                train_sizes=train_sizes, random_state=random_state)
            
        train_scores = 1 - train_scores
        test_scores = 1 - test_scores
            
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        
            #plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
            #                 train_scores_mean + train_scores_std, alpha=0.1,
            #                 color="r")
            #plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
            #                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', 
                     label="Training error_ k:"+ str(k))
        plt.plot(train_sizes, test_scores_mean, 'o-', 
                     label="Cross-validation error_k:"+ str(k), ls= "dotted")

plt.legend(loc="best")
plt.grid()


# fit the model by KNN
classifier = KNeighborsClassifier(n_neighbors=3, weights='distance')
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




