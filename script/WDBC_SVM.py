# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn import metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC

def get_scores(classifier, train_X, train_y, test_X, test_y):
    
    start_time = time.time()
    classifier.fit(train_X, train_y)
    prediction = classifier.predict(test_X)
    total_time = time.time() - start_time

    accuracy = metrics.accuracy_score(test_y, prediction)

    return accuracy, total_time



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

# Model Complexity experiments--kernel rbf with different gamma
title = "WDBC: Model Complexity (SVM)"
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
gamma = 0.0001 # based on above experiment
classifier = SVC(gamma = gamma, random_state=random_state)
start_time = time.time()
classifier.fit(train_X, train_y)
prediction = classifier.predict(test_X) #Predict with test set
total_time = time.time() - start_time
print("Accuracy on training set (rbf): "+ str(classifier.score(train_X, train_y))) # score on train set
print("Run time (rbf): " + str(total_time) + " sec") # Run time

# fit the model by SVC (linear)
classifier = SVC(kernel = 'linear', random_state=random_state)
start_time = time.time()
classifier.fit(train_X, train_y)
prediction = classifier.predict(test_X) #Predict with test set
total_time = time.time() - start_time
print("Accuracy on training set (linear): "+ str(classifier.score(train_X, train_y))) # score on train set
print("Run time (linear): " + str(total_time) + " sec") # Run time



# Model Complexity experiments--kernel poly with different degree ==> greyed out as it took too much time to run
#title = "WDBC: Model Complexity (SVM)"
#kernel = 'poly'
#degree = np.arange(3,6,1)

#train_scores, test_scores = validation_curve(SVC(random_state=random_state,
#                                                 kernel = kernel), 
#                                             train_X, train_y, 
#                                             param_name="degree",
#                                             param_range=degree, cv=cv)
#train_scores = 1 - train_scores
#test_scores = 1 - test_scores

#plt.figure()
#plt.title(title)
#plt.plot(degree, train_scores.mean(axis=1), 'b', label="Training error")
#plt.plot(degree, test_scores.mean(axis=1), 'g', label="Cross-validation error")
#plt.ylabel('Error')
#plt.xlabel('Degree of kernel')
#plt.legend(loc="best")
#plt.grid() 





# Learning curve experiments (compare with rbf and linear)
title = "WDBC: Learning Curves (SVM)"
n_jobs = -1
train_sizes = np.linspace(.1, 1.0, 10)
gamma = 0.0001 # based on above experiment

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
                SVC(kernel = 'linear', random_state=random_state), 
                train_X, train_y, cv=cv, n_jobs=n_jobs, 
                train_sizes=train_sizes, random_state=random_state)
            
train_scores = 1 - train_scores
test_scores = 1 - test_scores
            
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', 
                     label="Training error (linear)", color = "g")
plt.plot(train_sizes, test_scores_mean, 'o-', 
                     label="Cross-validation error (linear)", color = "g", ls= "dotted")


plt.legend(loc="best")
plt.grid()







# fit the final model by SVC(linear)
classifier = SVC(kernel = 'linear', random_state=random_state)
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


