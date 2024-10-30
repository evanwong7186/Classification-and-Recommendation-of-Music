import itertools
import numpy as np
import pandas as pd
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

for feature in combinations:
    X = data[list(feature)]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    print(f'Features: {feature}')
    print(f'Accuracy: {accuracy}')
    conf_matrix_df = pd.DataFrame(conf_matrix, 
                                  index=label_encoder.classes_, 
                                  columns=label_encoder.classes_)
    print('Confusion Matrix:')
    print(conf_matrix_df)
    print('\n' + '-'*50 + '\n')
