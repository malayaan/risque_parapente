import numpy as np
import csv
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Read the CSV file, preprocess data, and create train and test sets (same as before)
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

# Define the parameter grid for the hyperparameters to be optimized
param_grid = {
    'n_estimators': range(10, 60, 2),
    'max_depth': range(1, 10, 2),
    'min_samples_split': range(2, 10, 2)
}

# Initialize the GridSearchCV object with the RandomForestClassifier and the parameter grid
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, scoring='accuracy', cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, Y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Get the best estimator from the GridSearchCV object
best_rf = grid_search.best_estimator_

# Calculate the training accuracy
train_accuracy = accuracy_score(Y_train, best_rf.predict(X_train))

# Calculate the test accuracy
test_accuracy = accuracy_score(Y_test, best_rf.predict(X_test))

# Print the optimal hyperparameters, training accuracy, and test accuracy
print("Optimal Hyperparameters:", best_params)
print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
