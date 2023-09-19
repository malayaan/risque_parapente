import numpy as np
import csv
import random
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Specify the path of the CSV file
csv_file_path = "C:/Users/decroux paul/Documents/mission R & D/bdd_model_previsionnel_mediane_gravite.csv"

# Read the CSV file
with open(csv_file_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    # Skip the first line
    next(csvreader)
    # Create empty lists for storing data
    X = []
    Y = []

    # Loop through the rows in the CSV file
    for line in csvreader:
        # Split each string on the semicolon and convert the resulting strings to integers
        line= [[float(num_str) for num_str in s.split(';')] for s in line]
        line = [num for sublist in line for num in sublist]
        # Append all columns except the last one to the data list
        X.append(line[:-1])
        # Append the last column to the last_column list
        Y.append(line[-1])
 
# Define the proportion of data to include in the test set
alpha = 2/3

# Calculate the number of elements to include in the test set
n_test = int(len(X) * alpha)

# Shuffle the X and Y lists randomly
data = list(zip(X, Y))
random.shuffle(data)
X, Y = zip(*data)

# Divide the lists into training and test sets
X_train, Y_train = X[:n_test], Y[:n_test]
X_test, Y_test = X[n_test:], Y[n_test:]

# Hyperparameters to optimize

min_samples_split= 4
n_estimators = 20
max_depth = 3

# Create a Random Forest model with the current hyperparameter values
rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)

# Train the model on the training data
rf.fit(X_train, Y_train)

# Evaluate the performance of the model on the test data
accuracy = rf.score(X_test, Y_test)

# Ask the user to enter a new element X to predict the gravity
new_X = []
for i in range(len(X[0])):
    new_value = input(f"Enter the value for feature {i+1}: ")
    new_X.append(float(new_value))

# Predict the gravity of the new element
predicted_gravity = rf.predict([new_X])
print("accuracy", accuracy)
print("Predicted gravity:", predicted_gravity[0])