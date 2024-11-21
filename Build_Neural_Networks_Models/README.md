# Build Neural Networks Models

This project demonstrates the power of neural networks in solving a variety of machine learning problems, including regression, classification, and dimensionality reduction. Through carefully crafted Jupyter notebooks, the project showcases practical implementations of Multi-Layer Perceptrons (MLPs) and Restricted Boltzmann Machines (RBMs) on diverse datasets.

Each notebook addresses a unique challenge, from predicting weight changes using calorie and exercise data to classifying LEGO bricks based on images. The project also explores text classification with TF-IDF vectorization and reduces high-dimensional data using RBMs to improve model performance.

## Notebooks Overview

### 1. [MLP Regression on Diet and Exercise Data](01.MLPRegression_diet.ipynb)
This notebook demonstrates how to predict weight changes using a neural network regression model.

- **Overview**:
  - Explore the data to understand key features.
  - Standardize the dataset for better model performance.
  - Build and train a regression model using a Multi-Layer Perceptron (MLP).
  - Evaluate the model to check prediction accuracy.
- **Dataset**: [2018 Calorie, Exercise, and Weight Changes](https://www.kaggle.com/datasets/chrisbow/2018-calorie-exercise-and-weight-changes)

---

### 2. [MLP Classification for Lower Back Pain Symptoms](02.MLPClassification_LowerBackPain.ipynb)
This notebook applies a neural network to classify lower back pain symptoms based on feature data.

- **Overview**:
  - Prepare and explore the data for classification.
  - Build and train a classification model using an MLP.
  - Evaluate the model to determine its effectiveness in diagnosing symptoms.
- **Dataset**: [Lower Back Pain Symptoms Dataset](https://www.kaggle.com/datasets/sammy123/lower-back-pain-symptoms-dataset)

---

### 3. [MLP Classification on 20 Newsgroups Dataset](03.MLPClassification_TestNewGroups.ipynb)
This notebook tackles text classification by categorizing newsgroup messages into different topics.

- **Overview**:
  - Explore the dataset to understand the structure of text data.
  - Convert text into numerical features using TF-IDF vectorization.
  - Build a prototype classification model using an MLP.
  - Experiment with additional models and evaluate their performance.
- **Dataset**: [20 Newsgroups](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)

---

### 4. [MLP Classification on LEGO Brick Images](04.MLPClassification_LegoImages.ipynb)
This notebook focuses on classifying LEGO bricks into different categories using image data.

- **Overview**:
  - Load and preprocess the LEGO brick image dataset.
  - Train a neural network classifier to recognize different brick types.
  - Evaluate the model's accuracy in classifying the bricks.
- **Dataset**: [LEGO Brick Images](https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images)

---

### 5. [RBM for Dimensionality Reduction on MNIST](05.RBM_DimReductions.ipynb)
This notebook explores dimensionality reduction with Restricted Boltzmann Machines (RBMs) and applies it to improve classification tasks.

- **Overview**:
  - Train a classifier using the original MNIST dataset.
  - Apply RBMs to reduce the dataset's dimensionality.
  - Train a new classifier on the reduced data and compare performance.
- **Dataset**: [MNIST 784](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

---

## Requirements

- Python 3.x
- Jupyter Notebook
- Libraries:
  - `scikit-learn`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `opencv`

---
## Datasets

- [2018 Calorie, Exercise, and Weight Changes](https://www.kaggle.com/datasets/chrisbow/2018-calorie-exercise-and-weight-changes)
- [Lower Back Pain Symptoms Dataset](https://www.kaggle.com/datasets/sammy123/lower-back-pain-symptoms-dataset)
- [LEGO Brick Images](https://www.kaggle.com/datasets/joosthazelzet/lego-brick-images)

## Key Learning 
- Build and evaluate neural networks for both regression and classification tasks.
- Preprocess data for machine learning, including standardization and feature extraction.
- Use MLPs for various real-world datasets, including text and image data.
- Apply RBMs for dimensionality reduction to improve classification performance.