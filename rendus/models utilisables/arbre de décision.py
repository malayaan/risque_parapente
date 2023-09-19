import numpy as np
import csv
import random
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz
import tkinter as tk
from tkinter import filedialog, simpledialog

# Create a GUI for browsing for the CSV file
root = tk.Tk()
root.withdraw()
csv_file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

# Read the CSV file
with open(csv_file_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    # Get the first row to extract column names
    column_names = next(csvreader)
    column_names = column_names[0]
    column_names = column_names.split(";")
    scolumn_names = [s.strip() for s in column_names]
    X = []
    Y = []

    for line in csvreader:
        line = [[float(num_str) for num_str in s.split(';')] for s in line]
        line = [num for sublist in line for num in sublist]
        X.append(line[:-1])
        Y.append(line[-1])

# Divide the lists into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Get the user's desired depth for the decision tree
desired_depth = int(input("Enter the desired depth of the decision tree: "))

# Create a decision tree with the specified depth
dt = DecisionTreeClassifier(max_depth=desired_depth, min_samples_split=2, min_samples_leaf=1)
dt.fit(X_train, Y_train)

# Make predictions using the test set
Y_pred = dt.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy score: ", accuracy)

colors = {
    '0': '#FBB4AE',
    '1': '#B3CDE3',
    '2': '#CCEBC5',
    '3': '#DECBE4',
}

# Export the decision tree to DOT format
dot_data = export_graphviz(dt, out_file=None, 
                           feature_names=column_names[:-1],
                           class_names=['0','1', '2', '3'], 
                           filled=True, rounded=True, special_characters=False,
                           proportion=False, impurity=False, node_ids=False)

import re

def remove_value_info(dot_data):
    return re.sub(r'\\nvalue = .*?\\n', '\\n', dot_data)

def remove_samples_info(dot_data):
    return re.sub(r'\\nsamples = [0-9]+', '', dot_data)

def remove_class_info(dot_data):
    # Get all node ids in the DOT data
    node_ids = re.findall(r'(\d+) \[label=', dot_data)

    # Identify leaf nodes by checking if they have "->" in their line
    leaf_nodes = [node_id for node_id in node_ids if f"{node_id} ->" not in dot_data]

    # Remove class information from non-leaf nodes
    for node_id in node_ids:
        if node_id not in leaf_nodes:
            dot_data = re.sub(fr'({node_id} \[label=")([^"]*)\\nclass = [^"]*("[^]]*\])', r'\1\2\3', dot_data)

    return dot_data

modified_dot_data = remove_value_info(dot_data)

modified_dot_data_no_samples = remove_samples_info(modified_dot_data)

modified_dot_data_no_class = remove_class_info(modified_dot_data_no_samples)

# Create the visualization
graph = graphviz.Source(modified_dot_data_no_class)
graph.render('decision_tree')

# Display the decision tree in Jupyter Notebook
graph.format = 'png'
graph.render('decision_tree', view=True)
