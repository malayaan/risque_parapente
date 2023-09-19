import numpy as np
import csv
import random
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import graphviz

# Specify the path of the CSV file
csv_file_path = "C:/Users/decroux paul/Documents/mission R & D/bdd model prévisionnel - données complétées -  gravité.csv"

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
        line = [[float(num_str) for num_str in s.split(';')] for s in line]
        line = [num for sublist in line for num in sublist]
        # Append all columns except the last one to the data list
        X.append(line[:-1])
        # Append the last column to the last_column list
        Y.append(line[-1])
        
# Divide the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Define the hyperparameters to optimize
param_grid = {
    'max_depth': [2,3,4,5,6,7,8,9,10,15,20,25,30,35],
    'min_samples_split': [2,3,4,5,6,7,8,9,10],
    'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10]
}

# Create an instance of the DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)

# Create a GridSearchCV object with the hyperparameters to optimize
grid_search = GridSearchCV(dt, param_grid, cv=5)

# Train the GridSearchCV object on the training data
grid_search.fit(X_train, Y_train)

# Print the optimal hyperparameters and the corresponding training score
print("Optimal hyperparameters: ", grid_search.best_params_)
print("Training score: ", grid_search.best_score_)

# Use the optimal hyperparameters to predict labels for the test data
Y_pred = grid_search.predict(X_test)

# Print the accuracy score on the test data
accuracy = accuracy_score(Y_test, Y_pred)
print("Test accuracy: ", accuracy)

# Afficher la fréquence de la classe la plus nombreuse
max_class = max(Y_train, key=list(Y_train).count)
max_freq = list(Y_train).count(max_class) / len(Y_train)
print("Fréquence de la classe la plus nombreuse :", max_freq)