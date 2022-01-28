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

# %%Training the model
model = LogisticRegression(max_iter=10000)

model.fit(X_train, Y_train)

# %%Evaluating the model

# testing the accuracy on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy of the Training data is ', training_data_accuracy)

# testing the accuracy on the testing data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy of the Testing data is ', test_data_accuracy)


# %%running the model on an example
def predict_output(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The Person does not have a Heart Disease'
    else:
        return 'The Person has Heart Disease'


patient1 = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
patient2 = (37, 0, 1, 120, 263, 0, 1, 173, 0, 0, 2, 0, 3)

print(predict_output(patient1))
print(predict_output(patient2))
