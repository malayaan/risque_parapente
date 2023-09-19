# Importation des bibliothèques nécessaires
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import csv

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
        print(X[-1])
        # Append the last column to the last_column list
        Y.append(line[-1])

# Conversion de la liste de liste en DataFrame Pandas
X = pd.DataFrame(X, columns=['pph age',	'pph altitude',	'ac altitude',	'pph plaine 0 montagne 1',	'ac plaine 0 montagne 1',	'different relief ac pph',	'pph distance',	'PPH SEXE',	'PPH HANDICAPE',	'PPH NB HEURES PRATIQUE',	'AC EXPERIENCE',	'AC FREQUEE',	'AC BREVET VOL VOL LIBRE',	'AC NIVEAU PROGRESSION VOL LIBRE',	'AC ECOLE VOL LIBRE',	'AC 0 BI VOL LIBRE',	'AC TYPE VOL VOL LIBRE',	'AC SITE CONNAISSAE VOL LIBRE',	'AC TYPE AILE VOL LIBRE',	'AC SURFACE AILE',	'AC AILE HOMOLOGATION PARAPENTE',	'AC PARACHUTE PRESEE VOL LIBRE',	'AC SELLETTE PROTECTION PARAPENTE',	'AC CASQUE PRESEE VOL LIBRE',	'AC CONDITION METEO VOL LIBRE',	'AC NOMBRE DE JOUR DEPUIS LE D2BUT DE lANNEE',	'AC HEURE',	'Saison'])

# Conversion de la liste d'entier en série Pandas
Y = pd.Series(Y)

# Création de l'objet Random Forest
rf = RandomForestClassifier()

# Entraînement du modèle
rf.fit(X, Y)

# Calcul de la variable importance
importance = pd.Series(rf.feature_importances_, index=X.columns)

# Affichage de la variable importance triée par ordre décroissant
print(importance.sort_values(ascending=False))
