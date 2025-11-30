import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression

# Load and preprocess Exammarks dataset
exam = pd.read_csv("C:\\Users\\prade\\Desktop\\PYTHON_FILES\\KT programs\\Datanew\\Datanew\\Exammarks.csv")

exam['hours'].fillna(exam['hours'].median(), inplace=True)

X_exam = exam.drop('marks', axis=1)
y_exam = exam['marks']  # Fix: Select only 'marks' column as target

# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_exam, y_exam)

# Prediction function
def predict(hours, age, internet):
    input_data = np.array([[hours, age, internet]])
    return lr.predict(input_data)

print("Predicted Marks:", predict(5, 7, 6))
