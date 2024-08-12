# Credit Card Fraud Detection

This repository contains the code and documentation for a machine learning project focused on detecting fraudulent credit card transactions. The project utilizes various machine learning techniques to build and evaluate models that can accurately identify fraudulent behavior.

## Project Overview

Credit card fraud detection is a crucial task for financial institutions, aiming to identify suspicious transactions and prevent financial losses. This project explores different classification algorithms to build a predictive model that effectively distinguishes between fraudulent and legitimate transactions.

## Files in This Repository

- **Credit Card classification model.ipynb**: The Jupyter notebook containing the code for data preprocessing, feature engineering, model training, and evaluation.
- **fraudTest.csv** : The dataset used for this project.

## Steps in the Project

### 1. Data Preprocessing
- **Loading Data**: The dataset is loaded, and an initial exploratory data analysis (EDA) is performed to understand the data structure and distribution.
- **Data Cleaning**: Missing values, outliers, and inconsistencies in the dataset are handled to ensure the quality of the input data.
- **Feature Engineering**: Key features were created or transformed, such as transaction time intervals, spending patterns, and anomaly scores, to improve model performance.

### 2. Model Building
- **Model Selection**: Several classification models were tested, including Naive Bayes, Decision Trees, Random Forest, and Logistic Regression.
- **Hyperparameter Tuning**: Techniques like Grid Search and Cross-Validation were employed to optimize model performance.
- **Model Evaluation**: The models were evaluated using metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### 3. Results and Findings.
- **Feature Importance**: Features such as transaction time, merchant category, and customer history were found to be most influential in predicting fraud.
- **Recommendations**: The model can be used in real-time fraud detection systems to flag suspicious transactions for further investigation.

### 4. Future Work-
**Further Optimization**: 
Experimenting with advanced ensemble methods and deep learning approaches to enhance model accuracy.


### Build
- Python 3.x
- Jupyter Notebook
- Libraries: pandas, scikit-learn, matplotlib, seaborn

