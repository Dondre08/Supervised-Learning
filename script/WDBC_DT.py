# -*- coding: utf-8 -*-

import time
import matplotlib.pyplot as plt
import numpy as np
import graphviz

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve



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

max_depth = 3 # based on the above result
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
title = "WDBC: Learning Curves (Decision Tree)"
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

# See what are the important features
classifier.feature_importances_ 

export_graphviz(classifier,out_file='tree.dot')

# Visualize decision tree
dot_data = export_graphviz(classifier, out_file=None, 
                         feature_names=breast.feature_names,  
                         class_names=breast.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  

graph = graphviz.Source(dot_data)
graph