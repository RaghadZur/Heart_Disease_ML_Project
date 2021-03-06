# %% importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# %%importing the data as a pandas dataframe
df = pd.read_csv("heart.csv")

# %%exploring the data
# printing the first five rows of the dataframe
print(df.head())

# getting info about the data
print(df.info())

# getting statistical measures about the data
print(df.describe())

# checking the distribution of the target variable
print(df['target'].value_counts())

# %%finding correlation among the attributes
plt.figure(figsize=(20, 10))

sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
df.hist(figsize=(10, 10), layout=(4, 4))

plt.show()

# %%Spliting the data
X = df.drop(columns='target', axis=1)
Y = df['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=40)

# %%Training the logistic regression model
lr_model = LogisticRegression(max_iter=10000)
lr_model.fit(X_train, Y_train)

# %%Evaluating logistic regression model

# testing the accuracy on the training data
X_train_prediction1 = lr_model.predict(X_train)
training_data_accuracy1 = accuracy_score(X_train_prediction1, Y_train)
print('Accuracy of the Training data when using LR is ', training_data_accuracy1)

# testing the accuracy on the testing data
X_test_prediction1 = lr_model.predict(X_test)
test_data_accuracy1 = accuracy_score(X_test_prediction1, Y_test)
print('Accuracy of the Testing data when using LR is ', test_data_accuracy1)

# %% training the decision trees classifier model
dtc_model = DecisionTreeClassifier()
dtc_model.fit(X_train,Y_train)

# %%Evaluating the decision trees classifier model

# testing the accuracy on the training data
X_train_prediction2 = dtc_model.predict(X_train)
training_data_accuracy2 = accuracy_score(X_train_prediction2, Y_train)
print('Accuracy of the Training data when using DTC is ', training_data_accuracy2)

# testing the accuracy on the testing data
X_test_prediction2 = dtc_model.predict(X_test)
test_data_accuracy2 = accuracy_score(X_test_prediction2, Y_test)
print('Accuracy of the Testing data when using DTC is ', test_data_accuracy2)

# %% training the random forest classifer model
rfc_model = RandomForestClassifier()
rfc_model.fit(X_train,Y_train)

# %% evaluating the random forest classifer model

# testing the accuracy on the training data
X_train_prediction3 = rfc_model.predict(X_train)
training_data_accuracy3 = accuracy_score(X_train_prediction3, Y_train)
print('Accuracy of the Training data when using RFC is ', training_data_accuracy3)

# testing the accuracy on the testing data
X_test_prediction = rfc_model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy of the Testing data when using RFC is ', test_data_accuracy)

# %%running the model on an example
def predict_output(input_data, model):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The Person does not have a Heart Disease'
    else:
        return 'The Person has Heart Disease'


patient1 = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
patient2 = (37, 0, 1, 120, 263, 0, 1, 173, 0, 0, 2, 0, 3)

print(predict_output(patient1, lr_model))
print(predict_output(patient2, lr_model))
