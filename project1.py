#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:36:36 2024

@author: alminagunduz
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jb

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix

#Step 1 Data Processing
df= pd.read_csv("Project_1_Data.csv")

#Step 2 Data Visualization
fig1 = plt.figure()
X = df['X'].values
Y = df['Y'].values
Z = df['Z'].values
Step = df['Step'].values

plt.plot(Step,X, label = 'X')
plt.plot(Step,Y, label = 'Y')
plt.plot(Step,Z, label = 'Z')

plt.xlabel('Step')
plt.ylabel("X,Y,Z Values")
plt.title('Values vs Step Plot')
plt.legend()

plt.show()

#Step 3 Correlation
fig2 = plt.figure()
corr_matrix = df.corr()
sns.heatmap(np.abs(corr_matrix))

#Step 4 Classification
# Assuming df is already defined and contains the 'Step' column.
y = df["Step"]  # Keep target variable 'Step' here.
X = df.drop(columns=["Step"])  # Drop 'Step' to create features DataFrame.
# Split the data.
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=888)
for train_index, test_index in split.split(X, y):
    strat_train_set = X.loc[train_index].reset_index(drop=True)
    strat_test_set = X.loc[test_index].reset_index(drop=True)
    train_y = y.loc[train_index].reset_index(drop=True)  
    test_y = y.loc[test_index].reset_index(drop=True)

# Scaling
scaler = StandardScaler()
scaler.fit(strat_train_set)  # Scale based on training data only to prevent data leak.
scaled_data_train = scaler.transform(strat_train_set)
scaled_data_train_df = pd.DataFrame(scaled_data_train, columns=strat_train_set.columns)
train_X = scaled_data_train_df  # Directly assign scaled DataFrame.
train_y = train_y  # Keep target variable as is.
scaled_data_test = scaler.transform(strat_test_set)
scaled_data_test_df = pd.DataFrame(scaled_data_test, columns=strat_test_set.columns)
test_X = scaled_data_test_df  # Directly assign scaled DataFrame.
test_y = test_y  # Keep target variable as is.

#Model 1: Support Vector Machine Classifier
m1 = SVC(random_state= 888)

params1 = {
    'C': [1,10,100],
    'kernel': ['linear','rbf','poly','sigmoid'],
    'gamma': ['scale','auto'],
}
print("\nRunning Grid Search CV for Support Vector Machine:")
grid_search = GridSearchCV(m1, params1, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_X, train_y)
best_params1 = grid_search.best_params_
print("Best Hyperparameters:", best_params1)
best_m1 = grid_search.best_estimator_

#Model 2: Random Forest Classifier
m2 = RandomForestClassifier(random_state = 888)

params2 = {
    'n_estimators': [10,50,100],
    'max_depth': [None,5,10,15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
print("\nRunning Grid Search CV for Random Forest:")
grid_search = GridSearchCV(m2, params2, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_X, train_y)
best_params2 = grid_search.best_params_
print("Best Hyperparameters:", best_params2)
best_m2 = grid_search.best_estimator_

#Model 3: Logistic Regression Classifier
m3 = LogisticRegression(random_state = 888)  

params3 = {
    'C':[1,2,3,4,5],
    'max_iter':[5000,6000,8000],
    'solver':['newton-cg','sag','saga']
}
print("\nRunning Grid Search CV for Logical Regression:")
grid_search = GridSearchCV(m3, params3, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_X, train_y)
best_params3 = grid_search.best_params_
print("Best Hyperparameters:", best_params3)
best_m3 = grid_search.best_estimator_

#Model 4: Randomized Search CV for KNeighbors Classifier
CV_model = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [1,2,3,4,5],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
print("\nRunning Randomized Search CV for KNeighborsClassifier:")
random_search = RandomizedSearchCV(CV_model, param_distributions=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
random_search.fit(train_X, train_y)
Y_pred_CV = random_search.predict(test_X)
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)
best_modelCV = random_search.best_estimator_

#Step 5: Performance

def getScores(true,pred):
    print("Precision score: ", precision_score(true, pred, average='macro'))
    print("Accuracy score: ", accuracy_score(true, pred))
    print("F1 score: ",f1_score(true, pred, average='weighted'))
    
    
    return None

#Model 1
best_m1.fit(train_X,train_y)
m1_pred = best_m1.predict(test_X)

print("\n~~Scores for Support Vector Machine Model:~~\n")
getScores(test_y,m1_pred)

#Model 2
best_m2.fit(train_X,train_y)
m2_pred = best_m2.predict(test_X)

print("\n~~Scores for Random Forest Model:~~\n")
getScores(test_y,m2_pred)

#Model 3
best_m3.fit(train_X,train_y)
m3_pred = best_m3.predict(test_X)

print("\n~~Scores for Logical Regression Model:~~\n")
getScores(test_y,m3_pred)

#Model 4
best_modelCV.fit(train_X, train_y)
CV_model_pred = best_modelCV.predict(test_X)

print("\n~~Scores for Randomized Search KNeighborsClassifier Model:~~\n")
getScores(test_y, CV_model_pred)

#Confusion Matrix for the Selected Model
fig3 = plt.figure()
cm = confusion_matrix(test_y, m1_pred)
sns.heatmap(cm, annot= True)
plt.title('Support Vector Machine Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show

#Step 6 Stacking Classifier
stacking_estimators = [('rf', best_m2), ('svm', best_m1)]

# The final estimator is the best Logistic Regression model from the grid search.
final_estimator = best_m3

# Create the StackingClassifier with the defined base learners and final estimator.
stack = StackingClassifier(estimators=stacking_estimators, final_estimator=final_estimator, cv=5)

# Fit the stacking classifier on the training data.
stack.fit(train_X, train_y)

# Make predictions on the test data.
stack_pred = stack.predict(test_X)

# Display performance metrics for the StackingClassifier.
print("\n~~Scores for Stacking Classifier:~~\n")
getScores(test_y, stack_pred)

# Confusion Matrix for Stacking Classifier
fig4 = plt.figure()
con_matrix_stack = confusion_matrix(test_y, stack_pred)
sns.heatmap(con_matrix_stack, annot=True)
plt.title('Stacking Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Step 7 Model Evaluation
print("\nSelected Model 1 as Joblib File:")
jb.dump(best_m1,"best_model.joblib")

real_data = [[9.375, 3.0625, 1.51],
        [6.995, 5.125, 0.3875],
        [0, 3.0625, 1.93],
        [9.4, 3, 1.8],
        [9.4, 3, 1.3]]
real_data = pd.DataFrame(real_data, columns=['X', 'Y', 'Z'])
scaled_r_data = scaler.transform(real_data)
j = pd.DataFrame(scaled_r_data, columns=real_data.columns)    

real_step = best_m1.predict(j)

print("Predicted Steps:",real_step)


        
