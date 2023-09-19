import numpy as np
import csv
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

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


# Define the number of trees in the random forest
n_estimators = 20

# Define the maximum depth of each tree in the random forest
max_depth = 3

# Define the minimum number of samples required to split a node in the random forest
min_samples_split = 4

# Define the number of folds
n_folds = 8

# Define the K-fold cross-validation iterator
kf = KFold(n_splits=n_folds)

# Create empty lists for storing the accuracy scores for each fold
accuracy_scores = []
train_accuracy_scores = []

# Loop through the folds in the cross-validation
for train_index, test_index in kf.split(X):
    # Split the data into training and test sets for this fold
    X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
    Y_train, Y_test = np.array(Y)[train_index], np.array(Y)[test_index]

    # Create a Random Forest model with the current hyperparameter values
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)

    # Train the model on the training data
    rf.fit(X_train, Y_train)

    # Evaluate the performance of the model on the training data for this fold
    train_accuracy = rf.score(X_train, Y_train)
    train_accuracy_scores.append(train_accuracy)

    # Evaluate the performance of the model on the test data for this fold
    accuracy = rf.score(X_test, Y_test)

    # Append the accuracy score to the list of scores for this fold
    accuracy_scores.append(accuracy)

# Calculate the mean and standard deviation of the accuracy scores across all folds
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)

# Print the mean and standard deviation of the accuracy scores
print("Mean accuracy: {:.3f}".format(mean_accuracy))
print("Standard deviation: {:.3f}".format(std_accuracy))
# Print the list of training accuracy scores for each fold
print("training accuracy:", np.mean(train_accuracy_scores))
# Print the list of accuracy scores for each fold
print("List of test accuracy scores for each fold:")
print(accuracy_scores)

