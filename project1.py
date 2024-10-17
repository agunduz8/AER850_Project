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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

#Step 1
df= pd.read_csv("Project_1_Data.csv")

#Step 2
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

