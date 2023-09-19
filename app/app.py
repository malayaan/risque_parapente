import numpy as np
import csv
import random
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, render_template, request

app = Flask(__name__)

def train_model(csv_file_path):
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=';')
        headers = next(csvreader)
        X = []
        Y = []

        for line in csvreader:
            line = [float(x) for x in line]
            X.append(line[:-1])
            Y.append(line[-1])

    alpha = 2/3
    n_test = int(len(X) * alpha)
    data = list(zip(X, Y))
    random.shuffle(data)
    X, Y = zip(*data)
    X_train, Y_train = X[:n_test], Y[:n_test]
    X_test, Y_test = X[n_test:], Y[n_test:]

    min_samples_split = 20
    n_estimators = 10
    max_depth = 10
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    rf.fit(X_train, Y_train)
    accuracy = rf.score(X_test, Y_test)

    return rf, headers[:-1], accuracy

# Replace "data.csv" with your own CSV file path
rf, feature_headers, accuracy = train_model("data.csv")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        new_X = [float(request.form[header]) for header in feature_headers]
        predicted_gravity = rf.predict([new_X])
        return render_template("index.html", feature_headers=feature_headers, accuracy=accuracy, predicted_gravity=predicted_gravity[0])

    return render_template("index.html", feature_headers=feature_headers, accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)
