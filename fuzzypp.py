import numpy as np
import sys
from numpy.random import seed
import pandas as pd

# set the random seeds to make sure results are reproducible
seed(12)

def kmeans_plus_plus_init(X, k):
    n_samples = X.shape[0]
    centers = [X[np.random.randint(n_samples)]]

    for _ in range(k - 1):
        dist_sq = np.array([min([np.sum((x-c)**2) for c in centers]) for x in X])
        probs = dist_sq/dist_sq.sum()
        cumprobs = probs.cumsum()
        r = np.random.rand()

        for j, p in enumerate(cumprobs):
            if r < p:
                centers.append(X[j])
                break

    return np.array(centers)

def fuzzy_cmeans(X, k, r, p):
    n_samples, n_features = X.shape

    # Initialize best variables
    best_J = float('inf')
    best_U = None
    best_centers = None

    for _ in range(r):
        centers = kmeans_plus_plus_init(X, k)

        U = np.random.rand(n_samples, k)
        U = U / U.sum(axis=1)[:, np.newaxis]

        prev_J = float('inf')

        while True:
            U_p = U ** p
            denominator = U_p.sum(axis=0)[:, np.newaxis]
            centers = np.dot(U_p.T, X) / denominator

            distances = np.zeros((n_samples, k))
            for i in range(k):
                diff = X - centers[i]
                distances[:, i] = np.sum(diff**2, axis=1)

            distances = np.maximum(distances, np.finfo(float).eps)

            exp = 2.0/(p-1)
            denominator = np.sum((1.0/distances) ** (exp/2), axis=1)

            for i in range(k):
                U[:, i] = (1.0/distances[:, i]) ** (exp/2) / denominator

            J = np.sum(U**p * distances)

            if abs(prev_J - J) < 1e-6:
                break

            prev_J = J

        if J < best_J:
            best_J = J
            best_U = U
            best_centers = centers

    return best_U, best_centers, best_J

def get_hard_clusters(U):
    return np.argmax(U, axis=1)

def calculate_hard_error(X, labels, centers):
    error = 0
    for i, x in enumerate(X):
        error += np.sum((x - centers[labels[i]])**2)
    return error

def main():
    if len(sys.argv) != 6:
        print("Usage: python3 fuzzypp.py inputdata k r p outputclusters")
        sys.exit(1)

    input_file = sys.argv[1]
    k = int(sys.argv[2])
    r = int(sys.argv[3])
    p = float(sys.argv[4])
    output_file = sys.argv[5]

    X = pd.read_csv(input_file, header=None).values

    U, centers, fuzzy_error = fuzzy_cmeans(X, k, r, p)

    labels = get_hard_clusters(U)

    hard_error = calculate_hard_error(X, labels, centers)

    np.savetxt(output_file, labels, fmt='%d', delimiter=',')

    print(f"Fuzzy Quantization Error (J): {fuzzy_error:.4f}")
    print(f"Hard Quantization Error: {hard_error:.4f}")

if __name__ == "__main__":
    main()

