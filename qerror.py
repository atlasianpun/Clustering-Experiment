import sys
import numpy as np
import pandas as pd

def compute_quantization_error(data, labels):
    unique_labels = np.unique(labels)
    quantization_error = 0.0

    for label in unique_labels:
        cluster_points = data[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        squared_distances = np.sum((cluster_points - centroid) ** 2)
        quantization_error += squared_distances

    return quantization_error

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 qerror.py <data_file> <labels_file>")
        sys.exit(1)

    data_file = sys.argv[1]
    labels_file = sys.argv[2]

    try:
        data = pd.read_csv(data_file, header=None).values
        labels = pd.read_csv(labels_file, header=None).values.flatten()

        if data.shape[0] != labels.shape[0]:
            print("Error: The number of data points and labels must match.")
            sys.exit(1)

        quantization_error = compute_quantization_error(data, labels)
        print(f"Quantization Error: {quantization_error:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()