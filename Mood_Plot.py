import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

data = pd.read_csv("data/data.csv")

features = ["valence", "loudness"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

kmeans = KMeans(n_clusters=4, random_state=42)
data["Cluster"] = kmeans.fit_predict(X_scaled)

mood_map = {
    0: "Optimistic", 
    1: "Melancholy", 
    2: "Anger",
    3: "Calm"
}
data['Mood'] = data["Cluster"].map(mood_map)

centroids = kmeans.cluster_centers_

centroids_df = pd.DataFrame(scaler.inverse_transform(centroids), columns=features)
centroids_df['Mood'] = [mood_map[i] for i in range(len(centroids))]

valence_loudness = px.scatter(data, x="valence", y="loudness", color='Mood',
                                  title="Mood Clustering",
                                  labels={"valence": "Valence", "loudness": "Loudness"})
valence_loudness.add_scatter(x=centroids_df["valence"], y=centroids_df["loudness"],
                                 mode='markers+text', text=centroids_df['Mood'],
                                 textposition='top center', marker=dict(symbol='x', size=15, color='black'))

valence_loudness.add_shape(type="line", x0=0.57, y0=-10,
                           x1=1, y1=-13,
                           line=dict(color="Black", width=3))

valence_loudness.add_shape(type="line", x0=0.38, y0=-15,
                           x1=0.57, y1=-10,
                           line=dict(color="Black", width=3))
valence_loudness.add_shape(type="line", x0=0.38, y0=-15,
                           x1=1, y1=-43,
                           line=dict(color="Black", width=3))
valence_loudness.add_shape(type="line", x0=0.57, y0=-10,
                           x1=0.525, y1=0,
                           line=dict(color="Black", width=3))
valence_loudness.add_shape(type="line", x0=0.38, y0=-15,
                           x1=0, y1=-13,
                           line=dict(color="Black", width=3))

valence_loudness.show()




