# %% importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %%importing the data as a pandas dataframe
df = pd.read_csv("heart.csv")

# %%printing the first five rows of the dataframe
print(df.head())

# getting info about the data
print(df.info())

# getting statistical measures about the data
print(df.describe())

# checking the distribution of the target variable
print(df['target'].value_counts())

# %%Spliting the data
X = df.drop(columns='target', axis=1)
Y = df['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=3)

#%%Training the model
model = LogisticRegression(max_iter=10000)

model.fit(X_train, Y_train)