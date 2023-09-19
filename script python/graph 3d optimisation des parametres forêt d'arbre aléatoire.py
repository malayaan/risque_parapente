import numpy as np
import csv
import random
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
# Hyperparameters to optimize
n_estimators_range = range(10, 60, 1) # Range of values for the number of trees
max_depth_range = range(1, 10, 1) # Range of values for the maximum depth
min_samples_split = 10 # Fixed value for the minimum number of samples required to split an internal node

# Lists to store the results
accuracy_list = []
n_estimators_list = []
max_depth_list = []

# Loop over the hyperparameter values to train and evaluate the model
for i in range(1):
    for n_estimators in n_estimators_range:
        for max_depth in max_depth_range:
            # Create a Random Forest model with the current hyperparameter values
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, )

            # Train the model on the training data
            rf.fit(X_train, Y_train)

            # Evaluate the performance of the model on the test data
            accuracy = rf.score(X_test, Y_test)

            # Add the results to the lists
            accuracy_list.append(accuracy)
            n_estimators_list.append(n_estimators)
            max_depth_list.append(max_depth)

# Convert the lists to numpy arrays
accuracy_array = np.array(accuracy_list)
n_estimators_array = np.array(n_estimators_list)
max_depth_array = np.array(max_depth_list)

# Reshape the arrays to 2D
accuracy_array = np.reshape(accuracy_array, (-1, len(max_depth_range)))
n_estimators_array = np.reshape(n_estimators_array, (-1, len(max_depth_range)))
max_depth_array = np.reshape(max_depth_array, (-1, len(max_depth_range)))

# Calculate the mean accuracy over the 15 runs
mean_accuracy_array = np.mean(accuracy_array, axis=0)

# Plot the 3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(n_estimators_array, max_depth_array, accuracy_array, c=accuracy_array.flatten(), cmap='viridis')
ax.set_xlabel('Number of Trees')
ax.set_ylabel('Max Depth')
ax.set_zlabel('Accuracy')
plt.show()