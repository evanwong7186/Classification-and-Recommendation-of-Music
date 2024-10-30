import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv("data/data_labeled.csv")

features = ['valence', 'loudness', 'speechiness', 'tempo']
y = data['Mood']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

combinations = itertools.combinations(features, 2)

k_values = range(5, 21)
accuracies = {k: [] for k in k_values}

for feature in combinations:
    X = data[list(feature)]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        
        knn.fit(X_train, y_train)
        
        y_pred = knn.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        accuracies[k].append(accuracy)
        
        print(f'Features: {feature}, K-value: {k}, Accuracy: {accuracy:.4f}')

average_accuracies = {k: np.mean(acc) for k, acc in accuracies.items()}

plt.figure(figsize=(10, 6))
plt.plot(list(average_accuracies.keys()), list(average_accuracies.values()), marker='o', linestyle='-', color='b')
plt.title('K-value vs Accuracy')
plt.xlabel('K-value')
plt.ylabel('Accuracy')
plt.xticks(list(average_accuracies.keys()))
plt.grid(True)
plt.show()
