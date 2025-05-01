import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Set the page configuration
st.set_page_config(page_title="K-Means Clustering on Iris Dataset", layout="wide")

# Title of the Streamlit app
st.title("🔍 K-Means Clustering on Iris Dataset")

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Sidebar for Exploratory Data Analysis (EDA)
st.sidebar.header("Exploratory Data Analysis")
if st.sidebar.checkbox("Show Dataset Overview"):
    st.subheader("Iris Dataset Overview")
    st.write(pd.DataFrame(X, columns=iris.feature_names).head())

    st.subheader("Feature Correlation")
    fig = plt.figure(figsize=(10, 6))
    sns.heatmap(pd.DataFrame(X, columns=iris.feature_names).corr(), annot=True, cmap='coolwarm', ax=plt.gca())
    st.pyplot(fig)

# Step 1: Normalize the Iris dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Implement K-Means from scratch
class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape
        np.random.seed(42)
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iter):
            self.labels = self._assign_clusters(X)
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, X):
        return self._assign_clusters(X)

# Sidebar for user inputs
st.sidebar.header("Cluster Parameters")
k_value = st.sidebar.slider('Number of Clusters (k)', 2, 10, 3)

# Apply the custom K-Means implementation
kmeans_s = KMeans(n_clusters=k_value)
kmeans_s.fit(X_scaled)
y_kmeans = kmeans_s.predict(X_scaled)

# # Display Cluster Centroids
# st.header("Cluster Centroids")
# st.write(kmeans_s.centroids)

# 2D Visualization using PCA
st.subheader("2D Visualization of Clusters using PCA")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', marker='o', alpha=0.7)
ax.scatter(pca.transform(kmeans_s.centroids)[:, 0], pca.transform(kmeans_s.centroids)[:, 1], s=300, c='red', marker='X')
ax.set_title('K-Means Clustering (2D Visualization using PCA)')
ax.set_xlabel('Sepal Width')
ax.set_ylabel('Sepal Length')
st.pyplot(fig)

# 3D Visualization using Plotly
st.subheader("Interactive 3D Visualization of Clusters")
fig = px.scatter_3d(x=X_scaled[:, 0], y=X_scaled[:, 1], z=X_scaled[:, 2],
                    color=y_kmeans.astype(str), symbol=y_kmeans.astype(str),
                    labels={'x': iris.feature_names[0], 'y': iris.feature_names[1], 'z': iris.feature_names[2]},
                    title="3D Scatter plot of K-Means Clustering")

# Add centroid markers to the plot
centroids_fig = go.Scatter3d(x=kmeans_s.centroids[:, 0], y=kmeans_s.centroids[:, 1], z=kmeans_s.centroids[:, 2],
                             mode='markers', marker=dict(size=10, color='red', symbol='x'), name='Centroids')
fig.add_trace(centroids_fig)
st.plotly_chart(fig)

# Step 4: Compare the cluster assignments with the actual class labels
st.subheader("Confusion Matrix")
cm = confusion_matrix(y, y_kmeans)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f'Cluster {i+1}' for i in range(k_value)], yticklabels=target_names, ax=ax)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted Clusters')
ax.set_ylabel('True Labels')
st.pyplot(fig)

# Step 5: Plot the silhouette scores to evaluate the quality of the clusters
st.subheader("Cluster Quality Metrics")

silhouette_avg = silhouette_score(X_scaled, y_kmeans)
db_score = davies_bouldin_score(X_scaled, y_kmeans)
ch_score = calinski_harabasz_score(X_scaled, y_kmeans)
sample_silhouette_values = silhouette_samples(X_scaled, y_kmeans)

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(sample_silhouette_values, kde=True, color='blue', ax=ax)
ax.set_title(f'Silhouette Scores (Avg: {silhouette_avg:.2f})')
ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Frequency')
st.pyplot(fig)

st.write(f"Average Silhouette Score for k={k_value}: **{silhouette_avg:.2f}**")
# st.write(f"Davies-Bouldin Score for k={k_value}: **{db_score:.2f}**")
# st.write(f"Calinski-Harabasz Score for k={k_value}: **{ch_score:.2f}**")

# Elbow Method for Optimal k
st.subheader("Elbow Method for Optimal k")
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_scaled)
    inertia.append(np.sum((X_scaled - kmeans.centroids[kmeans.predict(X_scaled)]) ** 2))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(k_range, inertia, 'bo-', markersize=8)
ax.set_xlabel('Number of clusters (k)')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method for Optimal k')
st.pyplot(fig)

# Final Thoughts
st.markdown("""
    ### Final Thoughts
    The K-Means algorithm provides a powerful way to group data into clusters. 
    However, the quality of clustering can vary depending on the choice of `k` and 
    the distribution of data. The combination of multiple cluster quality metrics 
    (like Silhouette Score can provide deeper insights into the cluster formation.
""")
