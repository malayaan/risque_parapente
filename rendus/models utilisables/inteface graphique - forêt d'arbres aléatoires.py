import numpy as np
import csv
import random
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import filedialog

# Create a GUI for browsing for the CSV file
root = tk.Tk()
root.withdraw()
csv_file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

# Read the CSV file
with open(csv_file_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=';')
    # Extract the headers of the CSV file
    headers = next(csvreader)
    # Create empty lists for storing data
    X = []
    Y = []

    # Loop through the rows in the CSV file
    for line in csvreader:
        # Convert the strings to floats
        line = [float(x) for x in line]
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
min_samples_split = 20
n_estimators = 10
max_depth = 10

# Create a Random Forest model with the current hyperparameter values
rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)

# Train the model on the training data
rf.fit(X_train, Y_train)

# Evaluate the performance of the model on the test data
accuracy = rf.score(X_test, Y_test)

# Create a GUI for entering the new feature values and displaying the predicted gravity
root = tk.Tk()
root.title("Enter Feature Values")

# Create a label for each feature and an input field for the user to enter the value
entries = []
for i, header in enumerate(headers[:-1]):
    tk.Label(root, text=header).grid(row=i, column=0)
    e = tk.Entry(root)
    e.grid(row=i, column=1)
    entries.append(e)

# Create a label for displaying the predicted gravity
result_label = tk.Label(root, text="")
result_label.grid(row=len(headers), column=0, columnspan=2)

# Create a label for displaying the accuracy of the model
accuracy_label = tk.Label(root, text=f"Accuracy: {accuracy:.2f}")
accuracy_label.grid(row=len(headers)+1, column=0, columnspan=2)

# Create a button for the user to submit the feature values
def predict_gravity():
    # Get the new feature values from the input fields
    new_X = []
    for entry in entries:
        new_X.append(float(entry.get()))

    # Predict the gravity of the new element
    predicted_gravity = rf.predict([new_X])

    # Update the label with the predicted gravity value
    result_label.configure(text=f"Predicted gravity: {predicted_gravity[0]}")


tk.Button(root, text="Predict Gravity", command=predict_gravity).grid(row=len(headers)+1, column=1)

root.mainloop()
