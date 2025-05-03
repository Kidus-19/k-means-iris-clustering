An interactive web application for exploring K-Means clustering with comprehensive visualization and analysis capabilities.

## Features

### Core Functionality
- Custom K-Means implementation from scratch
- Comparison with scikit-learn's KMeans
- Support for both Iris dataset and custom CSV uploads

### Visualization
- 2D and 3D cluster visualization using:
  - PCA
  - t-SNE
  - UMAP
- Interactive Plotly charts
- Parallel coordinates plot for cluster profiling

### Analysis Tools
- Feature selection interface
- Adjustable algorithm parameters:
  - Number of clusters (k)
  - Maximum iterations
  - Tolerance level
- Cluster quality metrics:
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Score

### Advanced Features
- Elbow method with automatic knee detection
- Silhouette analysis diagrams
- Cluster profile statistics
- Data export functionality

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/Kidus-19/k-means-iris.git
cd k-means-iris
