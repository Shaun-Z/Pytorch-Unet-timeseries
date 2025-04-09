import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import pandas as pd


def load_data(data_path):
    """Generates diverse synthetic time-series data with different patterns."""
    np.random.seed(42)
    data = pd.read_csv(data_path)# load dlc data
    
    scaler = TimeSeriesScalerMeanVariance()  # Normalize
    data_scaled = scaler.fit_transform(np.array(data))
    return data_scaled


# Step 2: Shape-Based Distance Function
def shape_based_distance(series1, series2):
    """Computes the shape-based distance using cross-correlation."""
    series1 = series1.ravel()
    series2 = series2.ravel()
    
    # Normalize both series
    series1 -= np.mean(series1)
    series2 -= np.mean(series2)

    # Compute cross-correlation
    cross_corr = np.correlate(series1, series2, mode='full')
    max_corr = np.max(cross_corr)

    # Shape-based distance
    return 1 - (max_corr / (np.linalg.norm(series1) * np.linalg.norm(series2)))


# Step 3: Adaptive k-Shape Clustering
def adaptive_kshape(data, threshold=0.4):
    """Adaptive k-Shape clustering using shape-based distance to determine the number of clusters."""
    
    K = 2  # Initial number of clusters
    cluster_labels = None

    while True:
        print(f"---update------ Clustering with {K} clusters...")
        kshape = KShape(n_clusters=K, random_state=42)
        cluster_labels = kshape.fit_predict(data)
        centroids = kshape.cluster_centers_

        violating_clusters = []
        for cluster_id in range(K):
            cluster_series = data[cluster_labels == cluster_id]
            centroid = centroids[cluster_id].ravel()

            # Check for violations using shape-based distance
            for series in cluster_series:
                dist = shape_based_distance(centroid, series)
                if dist > threshold:
                    violating_clusters.append(cluster_id)
                    break  # Stop checking this cluster if any violation occurs

        if not violating_clusters:
            break  # Stop if all clusters are within the threshold

        K += len(set(violating_clusters))  # Increase the number of clusters adaptively

    return K, cluster_labels


# Step 4: Multi-Subplot Cluster Visualization
def plot_clusters_subplots(data, labels, title="Final Clustering Results"):
    """Visualizes clustered time-series data with each cluster in a separate subplot."""
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    
    fig, axes = plt.subplots(num_clusters, 1, figsize=(12, 3 * num_clusters), sharex=True)
    
    if num_clusters == 1:
        axes = [axes]  # Ensure axes is iterable for a single cluster

    colors = plt.cm.get_cmap("tab10", num_clusters)  # Generate different colors

    for idx, label in enumerate(unique_labels):
        ax = axes[idx]
        cluster_series = data[labels == label]

        for series in cluster_series:
            ax.plot(series, alpha=0.4, color=colors(idx))  # Transparency for overlapping lines
        
        ax.plot(np.mean(cluster_series, axis=0), color="black", linewidth=2, label=f"Cluster {label}")  # Mean line

        ax.set_title(f"Cluster {label}")
        ax.set_ylabel("Electricity Consumption")
        ax.legend()

    plt.xlabel("Time")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# Run the Example
if __name__ == "__main__":
    data_path = './data/DLC_data/data_prepared_1/combined_dfx.csv' # dlc data_path
    data = load_data(data_path)
    K, labels = adaptive_kshape(data, threshold=0.9)
    print(f"Final number of clusters: {K}")
    plot_clusters_subplots(data, labels)
