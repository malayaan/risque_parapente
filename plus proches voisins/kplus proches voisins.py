import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import csv
from sklearn.model_selection import cross_val_score
import statistics

# Specify the path of the CSV file
csv_file_path = 'C:/Users/decroux paul/Documents/mission R & D/plus proches voisins niveau.csv'

# Read the CSV file
with open(csv_file_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    # Skip the first line
    next(csvreader)
    # Create empty lists for storing data
    X = []
    Y = []
    X_ex=[]
    
    # Loop through the rows in the CSV file
    for line in csvreader:
        # Split each string on the semicolon and convert the resulting strings to integers
        line= [[float(num_str) for num_str in s.split(';')] for s in line]
        line = [num for sublist in line for num in sublist]
        X.append(line[1:6])
        Y.append(line[6])
        X_ex.append(line[8:14])
        

    X_ex =X_ex[1:171]
    print(len(X), len(Y))
    
# Calcul de la moyenne et de l'écart type de la liste Y
mean_ex = statistics.mean(Y)
std_dev_ex = statistics.stdev(Y)

# Normalisation de la liste par colonne
for i in range(len(X[0])):
    column = [row[i] for row in X]  # Sélectionne la colonne i
    mean = statistics.mean(column)  # Calcul de la moyenne de la colonne
    std_dev = statistics.stdev(column)  # Calcul de l'écart type de la colonne
    for j in range(len(column)):
        X[j][i] = (X[j][i] - mean) / std_dev  # Normalisation de la valeur


# Normalisation de la liste par colonne
for i in range(len(X_ex[0])):
    column = [row[i] for row in X_ex]  # Sélectionne la colonne i
    mean = statistics.mean(column)  # Calcul de la moyenne de la colonne
    std_dev = statistics.stdev(column)  # Calcul de l'écart type de la colonne
    for j in range(len(column)):
        X_ex[j][i] = (X_ex[j][i] - mean) / std_dev  # Normalisation de la valeur

# Calcul de la moyenne et de l'écart type de la liste
mean = statistics.mean(Y)
std_dev = statistics.stdev(Y)

# Normalisation de la liste
normalized_list = [(x - mean) / std_dev for x in Y]


# Division des données en train et test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Optimisation de l'hyperparamètre k en utilisant la validation croisée
n_neighbors = list(range(1, 30, 2))
cv_scores = []
for k in n_neighbors:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-1 * scores.mean())

# Choix de la meilleure valeur de k
best_k = n_neighbors[cv_scores.index(min(cv_scores))]

# Entraînement du modèle avec la meilleure valeur de k
knn = KNeighborsRegressor(n_neighbors=best_k)
knn.fit(X_train, Y_train)

# prédiction des valeurs de Y pour les données X_ex
Y_ex = knn.predict(X_ex)
restored_list = [(x * std_dev_ex) + mean_ex for x in Y_ex]
# affichage des valeurs prédites
print("Les valeurs de Y prédites pour X_ex sont:", list(Y_ex))
"""
#concatene les listes
Y=Y+list(Y_ex)
X=X+X_ex
# Ouverture du fichier Excel en mode écriture

with open(csv_file_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    i=0
    # Boucle sur la liste de valeurs numériques à partir de l'index 3146
    for value in Y:
        # Écriture de la valeur dans la colonne C à la ligne correspondante
        writer.writerow([X[i],value])
        i+=1
"""
