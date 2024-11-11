# Build Classification Models

This repository Section contains a collection of Jupyter notebooks focused on exploring, analyzing, and building machine learning models using popular datasets and machine learning techniques. The primary dataset used for analysis and modeling in this project is the Titanic dataset, alongside a fashion image dataset (Fashion MNIST).

## Project Structure

### 1. [Exploring the Titanic Dataset](01.ExploringTheTitanicDataset.ipynb)
This notebook focuses on initial data exploration and visualization of the Titanic dataset. Key sections include:
   - **Exploring Data**: Analyze the dataset's structure, identify missing values, and investigate key statistics.
   - **Visualizing Relationships in Data**: Use visualizations (e.g., histograms, box plots, heatmaps) to understand relationships between features such as age, gender, class, and survival rate.
   - **Processing the Data**: Handle missing values, encode categorical features, and prepare the data for further analysis.

### 2. [Binary Classification using Logistic Regression](02.BinaryClassification_LR_Titanic.ipynb)
This notebook performs binary classification to predict Titanic survival using Logistic Regression. Key sections include:
   - **Train the Model**: Fit a Logistic Regression model to the Titanic dataset and interpret key coefficients.
   - **Evaluate Model Performance**: Calculate accuracy, precision, and recall to evaluate model performance and assess predictive power.

### 3. [Building Multiple Classification Models](03.MultipleClassificationModels.ipynb)
In this notebook, multiple classification models are built and compared to identify the best-performing model for the Titanic dataset. Key components include:
   - **Creating Helper Functions**: Implement functions for summarizing scores, building models, and comparing results.
   - **Model Comparison**: Train multiple models (e.g., Decision Tree, Random Forest, Logistic Regression) and compare performance metrics to choose the most effective model.

### 4. [Hyperparameter Tuning](04.HyperparameterTuning.ipynb)
This notebook focuses on hyperparameter tuning for various machine learning models to optimize performance. Models include:
   - **Logistic Regression, Decision Tree, and Random Forest**: Utilize Grid Search and Random Search to tune hyperparameters and find the best configuration for each model.

### 5. [Image Classification using Logistic Regression for Fashion MNIST Dataset](05.ImageClassification_LR_MNIST.ipynb)
This notebook tackles an image classification problem using the Fashion MNIST dataset. It employs Logistic Regression with `scikit-learn` to classify images into categories. Key sections include:
   - **Data Preparation**: Load and preprocess the Fashion MNIST dataset for classification.
   - **Model Training and Evaluation**: Train a Logistic Regression model and assess its accuracy in classifying fashion items.

## Requirements
- Python 3.x
- Jupyter Notebook
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- plotly

```bash
pip install numpy pandas matplotlib seaborn scikit-learn plotly

```
## Acknowledgments

[Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data) For the dataset used in the Titanic analysis.
[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)


## Kaggle Versions for the notebooks
- [Titanic 101: A Data Science Quest for Survival Insights ](https://www.kaggle.com/code/iamagamy/titanic-survival-secrets)