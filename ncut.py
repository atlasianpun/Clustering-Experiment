import sys
import numpy as np

def compute_ncut(W, labels):
    n = W.shape[0]
    unique_labels = np.unique(labels)
    ncut_value = 0.0

    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        non_cluster_indices = np.where(labels != label)[0]
        assoc_A_A = np.sum(W[np.ix_(cluster_indices, cluster_indices)])
        assoc_A_V = np.sum(W[cluster_indices, :])
        if assoc_A_V > 0:
            ncut_value += (1 - (assoc_A_A / assoc_A_V))

    return ncut_value

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 ncut.py W.csv iris-labels.csv")
        sys.exit(1)

    weight_matrix_file = sys.argv[1]
    labels_file = sys.argv[2]

    try:
        W = np.loadtxt(weight_matrix_file, delimiter=',')
    except Exception as e:
        print(f"Error loading weight matrix: {e}")
        sys.exit(1)

    try:
        labels = np.loadtxt(labels_file, delimiter=',', dtype=int)
    except Exception as e:
        print(f"Error loading labels: {e}")
        sys.exit(1)

    if W.shape[0] != W.shape[1] or W.shape[0] != len(labels):
        print("Error: Weight matrix and labels dimensions do not match.")
        sys.exit(1)

    ncut_value = compute_ncut(W, labels)
    print(f"Normalized Cut (ncut) Value: {ncut_value}")

if __name__ == "__main__":
    main()