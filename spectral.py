import numpy as np
import pandas as pd
from numpy.random import seed
import sys

seed(12)

def load_graph(filename):
    return pd.read_csv(filename, header=None).values

def create_laplacian(W):
    D = np.diag(np.sum(W, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(D), 1e-10)))
    L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    return L

def initialize_centroids(data, k):
    n = data.shape[0]
    centroids = np.zeros((k, data.shape[1]))
    idx = np.random.choice(n)
    centroids[0] = data[idx]

    for i in range(1, k):
        dist_sq = np.array([min([np.inner(data[j] - centroid, data[j] - centroid) for centroid in centroids[:i]]) for j in range(n)])
        probabilities = dist_sq / dist_sq.sum()
        cumulative_probabilities = probabilities.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids[i] = data[j]
                break

    return centroids

def assign_clusters(data, centroids):
    distances = np.array([[np.inner(data[i] - centroid, data[i] - centroid) for centroid in centroids] for i in range(len(data))])
    return np.argmin(distances, axis=1)

def update_centroids(data, labels, k):
    centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return centroids

def kmeanspp(data, k, r):
    best_labels = None
    best_error = np.inf
    for _ in range(r):
        centroids = initialize_centroids(data, k)
        for _ in range(100):
            labels = assign_clusters(data, centroids)
            new_centroids = update_centroids(data, labels, k)
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        error = quantization_error(data, labels, centroids)
        if error < best_error:
            best_error = error
            best_labels = labels
    return best_labels

def quantization_error(data, labels, centroids):
    error = 0
    for i in range(len(data)):
        centroid = centroids[labels[i]]
        error += np.sum((data[i] - centroid) ** 2)
    return error

def spectral_clustering(W, k):
    L = create_laplacian(W)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    U = eigenvectors[:, 1:k+1]
    U_norm = U / np.sqrt(np.sum(U * U, axis=1))[:, np.newaxis]
    clusters = kmeanspp(U_norm, k, r=10)
    return clusters

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 spectral.py graphinput k outputclusters")
        sys.exit(1)

    graph_file = sys.argv[1]
    k = int(sys.argv[2])
    output_file = sys.argv[3]
    W = load_graph(graph_file)
    clusters = spectral_clustering(W, k)
    np.savetxt(output_file, clusters, fmt='%d', delimiter=',')
    print("Created/Modified files during execution:")
    print(output_file)

if __name__ == "__main__":
    main()