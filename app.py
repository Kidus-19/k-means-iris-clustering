import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, silhouette_score, 
                           silhouette_samples, davies_bouldin_score, 
                           calinski_harabasz_score)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans as SKLearnKMeans
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from kneed import KneeLocator

# Constants
MAX_ITER = 300
TOLERANCE = 1e-4
RANDOM_STATE = 42

# Configuration
st.set_page_config(
    page_title="Advanced K-Means Clustering", 
    layout="wide",
    page_icon="🔍"
)

# Title and description
st.title("🔍 Advanced K-Means Clustering Tool")
st.markdown("""
    Explore clustering patterns in your data with this interactive K-Means analyzer.
    Visualize results in 2D/3D, compare with scikit-learn, and evaluate cluster quality.
""")

# Custom K-Means implementation
class KMeans:
    def __init__(self, n_clusters, max_iter=MAX_ITER, tol=TOLERANCE, random_state=RANDOM_STATE):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        
        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)
            
            new_centroids = np.array([X[self.labels == i].mean(axis=0) 
                                    for i in range(self.n_clusters)])
            
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

@st.cache_data
def load_default_data():
    iris = load_iris()
    return {
        'data': iris.data,
        'target': iris.target,
        'feature_names': iris.feature_names,
        'target_names': iris.target_names
    }

def handle_file_upload():
    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset (CSV)", 
        type=["csv"],
        help="Upload a CSV file with numerical data for clustering"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            if len(numeric_cols) == 0:
                st.error("No numeric columns found in the uploaded file")
                return None
            
            return {
                'data': df[numeric_cols].values,
                'feature_names': numeric_cols,
                'target': None,
                'target_names': None
            }
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    return None

def plot_2d_clusters(X_vis, y_kmeans, centroids_vis, feature_names, reducer_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y_kmeans, cmap='viridis', alpha=0.7)
    ax.scatter(centroids_vis[:, 0], centroids_vis[:, 1], s=300, c='red', marker='X', label='Centroids')
    ax.set_title(f'K-Means Clustering ({reducer_name} Visualization)')
    ax.set_xlabel(f'{reducer_name} Component 1')
    ax.set_ylabel(f'{reducer_name} Component 2')
    ax.legend()
    return fig

def create_3d_plot(X_vis, y_kmeans, centroids_vis, feature_names, reducer_name):
    fig = px.scatter_3d(
        x=X_vis[:, 0], y=X_vis[:, 1], z=X_vis[:, 2],
        color=y_kmeans.astype(str),
        labels={
            'x': f'{reducer_name} 1',
            'y': f'{reducer_name} 2', 
            'z': f'{reducer_name} 3'
        },
        title=f"3D {reducer_name} Visualization"
    )
    
    if centroids_vis is not None and centroids_vis.shape[1] >= 3:
        fig.add_trace(go.Scatter3d(
            x=centroids_vis[:, 0], y=centroids_vis[:, 1], z=centroids_vis[:, 2],
            mode='markers',
            marker=dict(size=10, color='red', symbol='x'),
            name='Centroids'
        ))
    return fig

def evaluate_clustering(X_scaled, y_true, y_pred, k_value, target_names):
    st.subheader("Cluster Evaluation")
    
    tab1, tab2, tab3 = st.tabs(["Metrics", "Confusion Matrix", "Silhouette Analysis"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Silhouette Score", f"{silhouette_score(X_scaled, y_pred):.2f}",
                     help="Higher values indicate better defined clusters (range: -1 to 1)")
        with col2:
            st.metric("Davies-Bouldin", f"{davies_bouldin_score(X_scaled, y_pred):.2f}",
                     help="Lower values indicate better clustering")
        with col3:
            st.metric("Calinski-Harabasz", f"{calinski_harabasz_score(X_scaled, y_pred):.2f}",
                     help="Higher values indicate better clustering")
    
    with tab2:
        if y_true is not None and target_names is not None:
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=[f'Cluster {i}' for i in range(k_value)],
                        yticklabels=target_names)
            ax.set_xlabel('Predicted Clusters')
            ax.set_ylabel('True Labels')
            st.pyplot(fig)
        else:
            st.warning("No ground truth labels available for confusion matrix")
    
    with tab3:
        fig, ax = plt.subplots(figsize=(10, 6))
        silhouette_vals = silhouette_samples(X_scaled, y_pred)
        y_lower = 10
        
        for i in range(k_value):
            cluster_silhouette_vals = silhouette_vals[y_pred == i]
            cluster_silhouette_vals.sort()
            y_upper = y_lower + cluster_silhouette_vals.shape[0]
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_vals,
                            alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * cluster_silhouette_vals.shape[0], str(i))
            y_lower = y_upper + 10
        
        silhouette_avg = silhouette_score(X_scaled, y_pred)
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax.set_title(f"Silhouette Plot (Avg: {silhouette_avg:.2f})")
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Cluster")
        st.pyplot(fig)

def show_cluster_profiles(X, feature_names, y_kmeans):
    st.subheader("Cluster Profiles")
    df = pd.DataFrame(X, columns=feature_names)
    df['Cluster'] = y_kmeans
    
    tab1, tab2 = st.tabs(["Statistics", "Parallel Coordinates"])
    
    with tab1:
        st.dataframe(
            df.groupby('Cluster').agg(['mean', 'std']).T,
            height=400
        )
    
    with tab2:
        fig = px.parallel_coordinates(
            df, 
            color='Cluster',
            dimensions=feature_names,
            color_continuous_scale=px.colors.diverging.Tealrose
        )
        st.plotly_chart(fig, use_container_width=True)

def elbow_method(X_scaled, max_k=10):
    st.subheader("Elbow Method for Optimal k")
    inertia = []
    k_range = range(1, max_k+1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X_scaled)
        inertia.append(np.sum((X_scaled - kmeans.centroids[kmeans.predict(X_scaled)]) ** 2))
    
    kl = KneeLocator(k_range, inertia, curve='convex', direction='decreasing')
    suggested_k = kl.elbow if kl.elbow is not None else 3
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(k_range, inertia, 'bo-')
    ax.vlines(suggested_k, min(inertia), max(inertia), colors='r', linestyles='dashed')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title(f'Suggested optimal k: {suggested_k}')
    st.pyplot(fig)
    
    return suggested_k

def main():
    # Data loading
    custom_data = handle_file_upload()
    if custom_data is not None:
        data = custom_data
    else:
        data = load_default_data()
    
    X = data['data']
    y = data['target']
    feature_names = data['feature_names']
    target_names = data['target_names']
    
    # Sidebar controls
    st.sidebar.header("Clustering Parameters")
    
    # Feature selection
    selected_features = st.sidebar.multiselect(
        "Select features for clustering",
        feature_names,
        default=feature_names
    )
    
    if selected_features:
        feature_indices = [feature_names.index(f) for f in selected_features]
        X = X[:, feature_indices]
        feature_names = [feature_names[i] for i in feature_indices]
    
    # Algorithm parameters
    k_value = st.sidebar.slider('Number of Clusters (k)', 2, 10, 3)
    max_iter = st.sidebar.slider("Maximum iterations", 100, 500, 300)
    tolerance = st.sidebar.slider("Tolerance", 1e-6, 1e-2, 1e-4, format="%e")
    
    # Visualization options
    st.sidebar.header("Visualization Options")
    reducer_type = st.sidebar.selectbox(
        "Dimensionality reduction method",
        ["PCA", "t-SNE", "UMAP"],
        index=0
    )
    
    # Data preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Clustering
    kmeans = KMeans(n_clusters=k_value, max_iter=max_iter, tol=tolerance)
    kmeans.fit(X_scaled)
    y_kmeans = kmeans.predict(X_scaled)
    
    # Dimensionality reduction
    if reducer_type == "PCA":
        reducer = PCA(n_components=3)
    elif reducer_type == "t-SNE":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=3, random_state=RANDOM_STATE)
    else:  # UMAP
        import umap
        reducer = umap.UMAP(n_components=3, random_state=RANDOM_STATE)
    
    X_vis = reducer.fit_transform(X_scaled)
    centroids_vis = reducer.transform(kmeans.centroids) if hasattr(reducer, 'transform') else None
    
    # Main display
    col1, col2 = st.columns(2)
    
    with col1:
        st.pyplot(plot_2d_clusters(
            X_vis[:, :2], y_kmeans, 
            centroids_vis[:, :2] if centroids_vis is not None else None,
            feature_names, reducer_type
        ))
    
    with col2:
        st.plotly_chart(create_3d_plot(
            X_vis, y_kmeans,
            centroids_vis,
            feature_names, reducer_type
        ), use_container_width=True)
    
    # Evaluation and analysis
    evaluate_clustering(X_scaled, y, y_kmeans, k_value, target_names)
    show_cluster_profiles(X_scaled, feature_names, y_kmeans)
    elbow_method(X_scaled)
    
    # Comparison with sklearn
    if st.checkbox("Compare with scikit-learn's KMeans"):
        sklearn_kmeans = SKLearnKMeans(
            n_clusters=k_value, 
            max_iter=max_iter,
            tol=tolerance,
            random_state=RANDOM_STATE
        )
        sklearn_kmeans.fit(X_scaled)
        y_sklearn = sklearn_kmeans.labels_
        
        st.subheader("Comparison with scikit-learn")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Custom Implementation Silhouette", 
                     f"{silhouette_score(X_scaled, y_kmeans):.2f}")
        
        with col2:
            st.metric("scikit-learn Silhouette", 
                     f"{silhouette_score(X_scaled, y_sklearn):.2f}")
    
    # Data export
    st.subheader("Export Results")
    output = pd.DataFrame({
        **{name: X[:,i] for i,name in enumerate(feature_names)},
        "Cluster": y_kmeans
    })
    
    if y is not None:
        output["True_Label"] = y
    
    csv = output.to_csv(index=False).encode()
    st.download_button(
        "Download cluster assignments as CSV",
        data=csv,
        file_name="cluster_results.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
