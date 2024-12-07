import sys
import numpy as np
import pandas as pd
from numpy.random import seed

seed(12)

def read_data(file_path):
    return pd.read_csv(file_path, header=None)

def initialize_centroids(data, k):
    n = data.shape[0]
    centroids = np.zeros((k, data.shape[1]))
    idx = np.random.choice(n)
    centroids[0] = data.iloc[idx]

    for i in range(1, k):
        dist_sq = np.array([min([np.inner(data.iloc[j] - centroid, data.iloc[j] - centroid) for centroid in centroids[:i]]) for j in range(n)])
        probabilities = dist_sq / dist_sq.sum()
        cumulative_probabilities = probabilities.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids[i] = data.iloc[j]
                break

    return centroids

def assign_clusters(data, centroids):
    distances = np.array([[np.inner(data.iloc[i] - centroid, data.iloc[i] - centroid) for centroid in centroids] for i in range(len(data))])
    return np.argmin(distances, axis=1)

def update_centroids(data, labels, k):
    centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return centroids

def kmeanspp(data, k, r):
    best_labels = None
    best_error = np.inf
    for _ in range(r):
        centroids = initialize_centroids(data, k)
        for _ in range(100):  # max iterations
            labels = assign_clusters(data, centroids)
            new_centroids = update_centroids(data, labels, k)
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        error = quantization_error(data, labels, centroids)
        if error < best_error:
            best_error = error
            best_labels = labels
    return best_labels, best_error

def quantization_error(data, labels, centroids):
    error = 0
    for i in range(len(data)):
        centroid = centroids[labels[i]]
        error += np.sum((data.iloc[i] - centroid) ** 2)
    return error

def main(inputdata, k, r, outputclusters):
    data = read_data(inputdata)
    k = int(k)
    r = int(r)
    labels, error = kmeanspp(data, k, r)
    np.savetxt(outputclusters, labels, fmt='%d', delimiter=',')
    print(f"Quantization Error: {error}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 kmeanspp.py <inputdata> <k> <r> <outputclusters>")
        sys.exit(1)
    main(*sys.argv[1:])