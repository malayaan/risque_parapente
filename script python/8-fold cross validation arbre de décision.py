import numpy as np
import csv
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

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
        
# Create a Random Forest model
rf = RandomForestClassifier()

# Define the hyperparameters to optimize
param_grid = {
    'max_depth': [4],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(rf, param_grid, cv=8)
grid_search.fit(X, Y)

# Print the optimal hyperparameters and corresponding training score
print("Optimal hyperparameters: ", grid_search.best_params_)
print("Training score: ", grid_search.best_score_)

# Calculate the test accuracy using the best model from the grid search
best_rf = grid_search.best_estimator_
scores = cross_val_score(best_rf, X, Y, cv=8)
print("Accuracy scores: ", scores)
print("Mean accuracy: ", np.mean(scores))
print("Standard deviation: ", np.std(scores))