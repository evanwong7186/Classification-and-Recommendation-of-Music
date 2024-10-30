import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

data = pd.read_csv("data/data.csv")


feature1 = input("Enter the first feature: ")
feature2 = input("Enter the second feature: ")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[[feature1, feature2]])


kmeans = KMeans(n_clusters=2, random_state=42)
data["Cluster"] = kmeans.fit_predict(X_scaled)


cluster_map = {0: "Class A", 1: "Class B"}
data['label'] = data["Cluster"].map(cluster_map)


fig = px.scatter(data, x=feature1, y=feature2, color='label', title="Clustering of Data")
fig.show()
fig.write_image("clustered_scatter_plot.png")


data.to_csv("labeled_data.csv", index=False)
