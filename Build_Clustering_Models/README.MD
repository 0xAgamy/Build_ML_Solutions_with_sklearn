# Build Clustering Models 

This project explores clustering techniques using both the Iris dataset and the MNIST 784 dataset. Through a series of Jupyter notebooks, we investigate various clustering algorithms, evaluate their performance, and fine-tune their hyperparameters. The project demonstrates practical applications of clustering, from simple models to more advanced approaches.

## Notebooks Overview

### 1. [Simple K-Means on Iris Dataset](01.SimpleKmean_Iris.ipynb)
This notebook introduces K-Means clustering on the Iris dataset, covering:
   - **Data Exploration with Visualizations**: Understanding feature distributions and relationships.
   - **Performing K-Means with 2 Features**: Visualizing clusters in 2D space.
   - **Performing K-Means with All Features**: Exploring clustering in higher dimensions.
   - **Cluster Evaluation Metrics**: Calculating scores to assess clustering performance:
     - Homogeneity Score
     - Completeness Score
     - V-Measure Score
     - Adjusted Rand Score
     - Adjusted Mutual Information Score
     - Silhouette Score

### 2. [Multiple Clustering Models on Iris Dataset](02.MultipleClusteringModels_iris.ipynb)
This notebook expands on clustering techniques by implementing a variety of algorithms:
   - **Helper Functions**: Streamline model training and evaluation.
   - **Clustering Algorithms**:
     - Agglomerative Clustering
     - DBSCAN
     - Mean-Shift Clustering
     - BIRCH
     - Affinity Propagation Clustering
     - Mini-Batch K-Means
     - Spectral Clustering

### 3. [Hyperparameter Tuning](03.HyperparametersTuning.ipynb)
This notebook focuses on optimizing the performance of key clustering models:
   - **K-Means**: Finding the optimal number of clusters and initialization strategies.
   - **DBSCAN**: Tuning `eps` and `min_samples` parameters.
   - **Mean-Shift**: Adjusting bandwidth to enhance clustering accuracy.

### 4. [Image Clustering on MNIST Dataset](04.ImageClustering.ipynb)
This notebook applies clustering algorithms to the MNIST 784 dataset:
   - **K-Means**: Cluster images of handwritten digits into groups.
   - **Mini-Batch K-Means**: A faster variant of K-Means for handling large datasets.

## Dataset Sources
- **Iris Dataset**: A classic dataset containing measurements of iris flowers, used for clustering demonstrations.
- **MNIST 784 Dataset**: A dataset of handwritten digits, applied to image clustering tasks. 

## Acknowledgments
- The Iris dataset is a classic dataset widely used in data science and machine learning.
- The MNIST dataset is a benchmark dataset for digit recognition and clustering.

