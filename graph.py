import sys
import numpy as np
import csv

def compute_weight_matrix(data, sigma):

    n = data.shape[0]
    weight_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                distance_squared = np.sum((data[i] - data[j]) ** 2)
                gamma_ij = distance_squared / (2 * sigma ** 2)
                weight_matrix[i, j] = np.exp(-gamma_ij)
            else:

                weight_matrix[i, j] = 1.0

    return weight_matrix

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 graph.py dataset sigma graphfile")
        sys.exit(1)

    dataset_file = sys.argv[1]
    sigma = float(sys.argv[2])
    output_file = sys.argv[3]

    try:
        data = np.loadtxt(dataset_file, delimiter=',')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    weight_matrix = compute_weight_matrix(data, sigma)

    try:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(weight_matrix)
        print(f"Weight matrix saved to {output_file}")
    except Exception as e:
        print(f"Error saving weight matrix: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()