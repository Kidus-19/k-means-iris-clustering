import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels_ = np.argmin(distances, axis=1)
            new_centroids = np.array([X[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

def main():
    st.set_page_config(layout="wide")
    st.title("Enhanced K-Means Clustering")
    
    # Data loading
    iris = load_iris()
    X = iris.data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Sidebar controls
    k = st.sidebar.slider("Number of clusters", 2, 5, 3)
    max_iter = st.sidebar.slider("Max iterations", 100, 500, 300)
    
    # Clustering
    kmeans = KMeans(n_clusters=k, max_iter=max_iter)
    kmeans.fit(X_scaled)
    
    # Evaluation
    silhouette = silhouette_score(X_scaled, kmeans.labels_)
    st.metric("Silhouette Score", f"{silhouette:.2f}")
    
    # Visualization
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_)
        ax.scatter(pca.transform(kmeans.centroids)[:, 0], 
                   pca.transform(kmeans.centroids)[:, 1], 
                   marker='x', s=200, color='r')
        st.pyplot(fig)
    
    with col2:
        fig = px.scatter_3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], 
                           color=kmeans.labels_.astype(str))
        st.plotly_chart(fig)
    
    # Comparison with sklearn
    if st.checkbox("Compare with sklearn"):
        sklearn_kmeans = SKLearnKMeans(n_clusters=k, max_iter=max_iter, random_state=42)
        sklearn_kmeans.fit(X_scaled)
        st.metric("sklearn Silhouette", f"{silhouette_score(X_scaled, sklearn_kmeans.labels_):.2f}")

if __name__ == "__main__":
    main()
