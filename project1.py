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

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

#Step 1
df= pd.read_csv("Project_1_Data.csv")

#Step 2
fig = plt.figure()
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

#Step 3
corr_matrix = df.corr()
sns.heatmap(np.abs(corr_matrix))



#Step 4

def extractor(df):
    # Extract the target variable (train_y) and features (df_X)
    y = df["Step"]
    X = df.drop(columns=["Step"])

    
    scaled_data = scaler.transform(X)
    scaled_data_df = pd.DataFrame(scaled_data, columns= X.columns)
    
    return scaled_data_df, y

X = df.drop(columns=["Step"])
scaler = StandardScaler()
scaler.fit(X)


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=501)
for train_index ,test_index in split.split(df,df["Step"]):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)


train_X,train_y = extractor(strat_train_set)
test_X,test_y = extractor(strat_test_set)


#model 1 Support vector machine
m1 = SVC(random_state= 501)

params1 = {
    'C': [1,10,100],
    'kernel': ['linear','rbf','poly','sigmoid'],
    'gamma': ['scale','auto'],
}

print("\nrunning grid search for SVM Model")
grid_search = GridSearchCV(m1, params1, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_X, train_y)
best_params1 = grid_search.best_params_
print("Best Hyperparameters:", best_params1)
best_m1 = grid_search.best_estimator_


#model 2 random forest
m2 = RandomForestClassifier(random_state = 501)

params2 = {
    'n_estimators': [10,50,100],
    'max_depth': [None,5,10,15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

print("\nrunning grid search for Random Forest Model")
grid_search = GridSearchCV(m2, params2, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_X, train_y)
best_params2 = grid_search.best_params_
print("Best Hyperparameters:", best_params2)
best_m2 = grid_search.best_estimator_


#model 3 logistic
m3 = LogisticRegression(random_state = 501)

params3 = {
    'C':[1,2,3,4,5],
    'max_iter':[5000,6000,8000],
    'solver':['newton-cg','sag','saga']
}
print("\nrunning grid search for Logi Model")
grid_search = GridSearchCV(m3, params3, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_X, train_y)
best_params3 = grid_search.best_params_
print("Best Hyperparameters:", best_params3)
best_m3 = grid_search.best_estimator_


#add RandomizedSearchCV
param_grid = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
    }

CV_model = RandomForestClassifier()
grid_search = RandomizedSearchCV(CV_model, param_distributions=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)
grid_search.fit(train_X, train_y)
Y_pred_CV = grid_search.predict(test_X)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
best_modelCV = grid_search.best_estimator_


accuracy_CV = accuracy_score(test_y, Y_pred_CV)
print(f"CV Accuracy: {accuracy_CV}")

#STEP 5: performance

def getScores(true,pred):
    print("Precision score: ", precision_score(true, pred, average= 'micro'))
    print("Accuracy score: ", accuracy_score(true, pred))
    print("F1 score: ",f1_score(true, pred, average= 'micro'))
    
    
    return None


#model 1
best_m1.fit(train_X,train_y)
m1_pred = best_m1.predict(test_X)

print("\n~~scores for SVM model~~\n")
getScores(test_y,m1_pred)


#model 2
best_m2.fit(train_X,train_y)
m2_pred = best_m2.predict(test_X)

print("\n~~scores for random forest model~~\n")
getScores(test_y,m2_pred)


#model 3
best_m3.fit(train_X,train_y)
m3_pred = best_m3.predict(test_X)

print("\n~~scores for logic model~~\n")
getScores(test_y,m3_pred)

cm = confusion_matrix(test_y, m2_pred)

disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot()

#Step 6 StackingClassifier
tb_RF_model = RandomForestClassifier(n_estimators=100, random_state=501)
tb_SVM_model = SVC(random_state= 501)

tb_RF_model.fit(train_X, train_y)
tb_SVM_model.fit(train_X, train_y)

fin_est = LogisticRegression()
stack = StackingClassifier(estimators=[('rf', tb_RF_model), ('svm',tb_SVM_model )], final_estimator=fin_est)

stack.fit(train_X,train_y)

Y_pred_stack = stack.predict(test_X)
accuracy_stack = accuracy_score(test_y, Y_pred_stack)
f1_stack = f1_score(test_y, Y_pred_stack, average='weighted')

con_matrix_stack = confusion_matrix(test_y, Y_pred_stack)

print(f"Stacking Classifier Accuracy: {accuracy_stack}")
print(f"Stacking F1 Score: {f1_stack}")

sns.heatmap(con_matrix_stack, annot= True)
plt.title('Stacking Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show


#STEP 7:
print("\ndumping model 2 into joblib file")
jb.dump(best_m2,"best_model.joblib")

real_data = [[9.375, 3.0625, 1.51],
        [6.995, 5.125, 0.3875],
        [0, 3.0625, 1.93],
        [9.4, 3, 1.8],
        [9.4, 3, 1.3]]
real_data = pd.DataFrame(real_data, columns=['X', 'Y', 'Z'])
scaled_r_data = scaler.transform(real_data)
j = pd.DataFrame(scaled_r_data, columns=real_data.columns)    

real_step = best_m2.predict(j)

print("Steps:",real_step)



        
