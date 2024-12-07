# Clustering Project

## Description

Implement various clustering algorithms to analyze the Iris dataset, ensuring reproducibility and accuracy.

## Table of Contents

- [Overview](#overview)
- [Random Seed](#random-seed)
- [Data Preparation](#data-preparation)
- [Implemented Programs](#implemented-programs)
  - [Part I](#part-i)
    - [qerror.py](#qerrorpy)
    - [kmeanspp.py](#kmeansppy)
    - [fuzzypp.py](#fuzzyppy)
  - [Part II](#part-ii)
    - [graph.py](#graphpy)
    - [ncut.py](#ncutpy)
    - [spectral.py](#spectralpy)
- [Research Recommendations](#research-recommendations)
- [Submission Guidelines](#submission-guidelines)
- [Community Standards](#community-standards)
- [Deadline](#deadline)

## Overview

This project implements various clustering algorithms, including k-means, fuzzy c-means, and spectral clustering, to analyze the Iris dataset. The goal is to evaluate the performance of these algorithms and ensure reproducibility of results by fixing the random seed.

## Random Seed

To ensure reproducibility, the random seed is set to **12** in all programs. This guarantees that the results remain consistent across multiple runs. The following lines are included at the top of each program:

```
# Set the random seeds to make sure results are reproduciblefrom numpy.random import seedseed(12)
```

## Data Preparation

The project uses 50% randomly selected instances from the Iris dataset. The dataset is provided in two files:

- `iris-data.csv`: Contains the data matrix
- `iris-labels.csv`: Contains the labels

To generate the 50% random subset, use `fraction_xy.py`:

```
python3 fraction_xy.py iris-data.csv iris-labels.csv 0.5 7
```

## Implemented Programs

### Part I

#### qerror.py

Evaluates the quantization error in clustering.

**Usage:**

```
python3 qerror.py iris-data.csv iris-labels.csv
```

#### kmeanspp.py

Implements the k-means++ algorithm for clustering.

**Usage:**

```
python3 kmeanspp.py inputdata k r outputclusters
```

#### fuzzypp.py

Implements fuzzy c-means with k-means++ initialization.

**Usage:**

```
python3 fuzzypp.py inputdata k r p outputclusters
```

### Part II

#### graph.py

Converts dataset into a complete graph with weighted edges.

**Usage:**

```
python3 graph.py dataset sigma graphfile
```

#### ncut.py

Evaluates the normalized cut error in clustering.

**Usage:**

```
python3 ncut.py W.csv iris-labels.csv
```

#### spectral.py

Implements spectral clustering (NJW) with k-means++.

**Usage:**

```
python3 spectral.py graphinput k outputclusters
```
