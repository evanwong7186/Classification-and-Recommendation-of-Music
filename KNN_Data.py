import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

data = pd.read_csv("data/data.csv")

features = ["valence", "loudness", "speechiness", "tempo"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

kmeans = KMeans(n_clusters=4, random_state=42)
data["Cluster"] = kmeans.fit_predict(X_scaled)

mood_map = {
    0: "Optimistic", 
    1: "Anger", 
    2: "Calm",
    3: "Melancholy"
}
data['Mood'] = data["Cluster"].map(mood_map)

data.to_csv("data/data_labeled.csv", index=False)

