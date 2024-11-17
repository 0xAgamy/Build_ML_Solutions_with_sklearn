# Build Regression Models

A series of Jupyter notebooks dedicated to building, testing, and tuning various regression models. Each notebook covers a different aspect of the regression modeling process, from data exploration to advanced model selection and hyperparameter tuning.

## Notebooks Overview

### 1. [Exploring the Automobile Dataset](01.Exploring_Automobile_Dataset.ipynb)
This notebook provides an initial examination of the Automobile dataset. Key sections include:
   - **Exploring Data**: Understanding the dataset’s structure, identifying key features, and handling missing values.
   - **Visualizing Relationships in Data**: Using visualizations (e.g., scatter plots, pair plots, heatmaps) to explore relationships between features such as horsepower, engine size, price, and fuel efficiency.

### 2. [Simple Regression with the Automobile Dataset](02.SimpleRegression_Automobile.ipynb)
This notebook demonstrates the process of building simple and multiple linear regression models:
   - **Building a Simple Regression Model with One Feature**: Select one feature to predict the target and evaluate its effectiveness.
   - **Trying Another Single Feature**: Evaluate another single-feature model and compare performance.
   - **Building a Multiple Linear Regression Model with Selected Features**: Use multiple features to predict the target variable.
   - **Using All Dataset Features**: Build and evaluate a regression model with all available features in the dataset.

### 3. [Simple Regression with Categorical Values (Exam Score Prediction)](03.SimpleRegression_CatValues_ExamScore.ipynb)
This notebook covers handling categorical data for regression tasks, with a focus on preparing data and avoiding common pitfalls:
   - **Standardizing Numeric Data**: Scaling numeric features for consistent model performance.
   - **Label Encoding and One-Hot Encoding**: Encoding categorical data for regression modeling. *'i called it Shaabalizition :)'*
   - **Linear Regression and the Dummy Trap**: Handling multicollinearity to avoid biased predictions.

### 4. [Multiple Regression with the Automobile Dataset](04.MultipleRegression_Automobile.ipynb)
In this notebook, we explore and build multiple regression models using various algorithms to find the best estimator for our problem:
   - **Defining Helper Functions**: Create reusable functions to build, train, and evaluate regression models efficiently.
   - **Testing Helper Functions**: Ensure our functions work correctly by testing them on different models.
   - **Building Models with Various Regression Techniques**:
       - **Lasso Regression**
       - **Ridge Regression**
       - **Elastic Net Regression**
       - **Support Vector Regression (SVR)**
       - **K-Nearest Neighbors (KNN) Regression**
       - **Stochastic Gradient Descent (SGD) Regression**
       - **Decision Tree Regression**

### 5. [Hyperparameter Tuning for Regression Models](05.HyperparameterTuning.ipynb)
This notebook focuses on optimizing model performance by tuning hyperparameters for four different regression models:
   - **Models Tuned**: Lasso, K-Nearest Neighbors, Decision Tree, and Support Vector Regression (SVR).
   - **GridSearchCV**: Utilize GridSearchCV to identify the best hyperparameter settings for each model to improve prediction accuracy and performance.

## Dataset
The dataset used in this project can be found on :
   - [Auto Mobile](https://www.kaggle.com/datasets/roger1315/automobiles).
   - [Exam Dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)


## Requirements
- Python 3.x
- Jupyter Notebook
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn


```bash
pip install numpy pandas matplotlib seaborn scikit-learn 

```