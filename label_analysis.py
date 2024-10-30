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

centroids = kmeans.cluster_centers_

centroids_df = pd.DataFrame(scaler.inverse_transform(centroids), columns=features)
centroids_df['Mood'] = [mood_map[i] for i in range(len(centroids))]

valence_loudness = px.scatter(data, x="valence", y="loudness", color='Mood',
                                  title="Clustering based on Valence and Loudness",
                                  labels={"valence": "Valence", "loudness": "Loudness"})
valence_loudness.add_scatter(x=centroids_df["valence"], y=centroids_df["loudness"],
                                 mode='markers+text', text=centroids_df['Mood'],
                                 textposition='top center', marker=dict(symbol='x', size=12, color='black'))
valence_loudness.show()

valence_speechiness = px.scatter(data, x="valence", y="speechiness", color='Mood',
                                     title="Clustering based on Valence and Speechiness",
                                     labels={"valence": "Valence", "speechiness": "Speechiness"})
valence_speechiness.add_scatter(x=centroids_df["valence"], y=centroids_df["speechiness"],
                                    mode='markers+text', text=centroids_df['Mood'],
                                    textposition='top center', marker=dict(symbol='x', size=12, color='black'))
valence_speechiness.show()

speechiness_tempo = px.scatter(data, x="speechiness", y="tempo", color='Mood',
                                   title="Clustering based on Speechiness and Tempo",
                                   labels={"speechiness": "Speechiness", "tempo": "Tempo"})
speechiness_tempo.add_scatter(x=centroids_df["speechiness"], y=centroids_df["tempo"],
                                  mode='markers+text', text=centroids_df['Mood'],
                                  textposition='top center', marker=dict(symbol='x', size=12, color='black'))
speechiness_tempo.show()
