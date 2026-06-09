# Semiconductor Failure Prediction using SVM

## Overview

This project implements a complete machine learning pipeline for predicting semiconductor manufacturing failures using the SECOM dataset from the UCI Machine Learning Repository. The primary objective is to identify defective semiconductor production instances using Support Vector Machines (SVM) while comparing two dimensionality reduction approaches:

1. **Principal Component Analysis (PCA)**
2. **Feature Selection using Random Forest Feature Importance**

The project demonstrates data preprocessing, handling missing values, class imbalance treatment, dimensionality reduction, hyperparameter optimization, and model evaluation.

---


# Theory of Support Vector Machine (SVM)

## Introduction

A **Support Vector Machine (SVM)** is a supervised machine learning algorithm primarily used for **classification** and **regression** tasks. It is particularly effective for high-dimensional datasets and aims to find the optimal decision boundary that separates different classes.

In semiconductor failure prediction, SVM is used to classify manufacturing instances as either:

* **Pass (-1)**
* **Fail (1)**

---

## Core Concept

The fundamental objective of SVM is to find the **best hyperplane** that separates data points belonging to different classes while maximizing the distance between the classes and the decision boundary.

Unlike many classification algorithms, SVM focuses on creating the largest possible margin between classes, which improves the model's ability to generalize to unseen data.

---

## Hyperplane

A **hyperplane** is the decision boundary that separates classes.

For a two-dimensional dataset, the hyperplane can be represented as:

[
w_1x_1 + w_2x_2 + b = 0
]

More generally:

[
w^T x + b = 0
]

where:

* (w) = weight vector
* (x) = feature vector
* (b) = bias term

The hyperplane divides the feature space into two regions:

[
w^T x + b > 0
]

and

[
w^T x + b < 0
]

---

## Margin

The **margin** is the distance between the hyperplane and the nearest data points from each class.

SVM attempts to maximize this margin:

[
\text{Margin} = \frac{2}{||w||}
]

A larger margin generally results in:

* Better generalization
* Improved robustness to noise
* Reduced overfitting

---

## Support Vectors

The data points closest to the hyperplane are known as **support vectors**.

These points are critical because they determine the location and orientation of the decision boundary.

Characteristics of support vectors:

* Directly influence the hyperplane
* Define the margin
* Removal of non-support vectors often has little impact on the model

The algorithm derives its name from these important observations.

---

## Optimization Objective

For linearly separable data, SVM solves the following optimization problem:

### Objective Function

[
\min \frac{1}{2} ||w||^2
]

### Subject To

[
y_i(w^T x_i + b) \geq 1
]

where:

* (x_i) = feature vector
* (y_i) = class label (+1 or -1)

Minimizing (||w||^2) effectively maximizes the margin.

---

## Soft Margin SVM

Real-world datasets are rarely perfectly separable.

To allow some classification errors, SVM introduces **slack variables** ((\xi_i)):

[
\min \frac{1}{2} ||w||^2 + C \sum \xi_i
]

where:

* (C) = regularization parameter
* (\xi_i) = penalty for misclassified points

### Role of C

#### Large C

* Strong penalty for errors
* Smaller margin
* Lower training error
* Greater risk of overfitting

#### Small C

* Allows some misclassifications
* Larger margin
* Better generalization
* Higher bias

In this project, the parameter (C) is optimized using GridSearchCV.

---

## Non-Linear Classification

Many datasets cannot be separated using a straight line or linear hyperplane.

In such cases, SVM uses a technique known as the **Kernel Trick** to transform data into a higher-dimensional space where separation becomes possible.

---

## Kernel Trick

The kernel trick enables SVM to perform complex nonlinear classification without explicitly computing higher-dimensional transformations.

Instead, it calculates similarities between data points using kernel functions.

### Common Kernel Functions

#### Linear Kernel

[
K(x_i, x_j) = x_i^T x_j
]

Used when data is approximately linearly separable.

---

#### Polynomial Kernel

[
K(x_i, x_j) = (x_i^T x_j + c)^d
]

Captures polynomial relationships between features.

---

#### Radial Basis Function (RBF) Kernel

[
K(x_i, x_j) = e^{-\gamma ||x_i - x_j||^2}
]

The RBF kernel is the most widely used kernel and can model highly complex decision boundaries.

This project uses the **RBF kernel** because semiconductor sensor data often exhibits nonlinear relationships.

---

## Gamma Parameter

The parameter (\gamma) controls the influence of individual training samples in the RBF kernel.

### Small Gamma

* Produces smoother decision boundaries
* Considers points farther away
* Lower variance

### Large Gamma

* Produces more complex boundaries
* Focuses on nearby points
* Higher variance
* Greater risk of overfitting

In this project, multiple gamma values are evaluated using GridSearchCV.

---

## Importance of Feature Scaling

SVM relies heavily on distance calculations.

Features with larger numerical ranges can dominate the learning process if scaling is not applied.

To address this issue, features are standardized using:

[
z = \frac{x - \mu}{\sigma}
]

where:

* (\mu) = feature mean
* (\sigma) = feature standard deviation

This transformation ensures that all features contribute equally during training.

---

## Why SVM is Suitable for Semiconductor Failure Prediction

The SECOM dataset possesses several characteristics that make SVM a strong choice:

* High-dimensional sensor measurements
* Complex nonlinear relationships
* Imbalanced failure occurrences
* Relatively limited failure samples

Advantages of SVM in this context include:

* Effective handling of high-dimensional data
* Strong generalization capability
* Robust performance with nonlinear kernels
* Ability to identify subtle patterns in sensor readings

---

## Advantages of SVM

* Effective in high-dimensional spaces
* Strong theoretical foundation
* Good generalization performance
* Works well with nonlinear decision boundaries
* Memory efficient due to reliance on support vectors

---

## Limitations of SVM

* Computationally expensive for large datasets
* Sensitive to hyperparameter selection
* Requires feature scaling
* Less interpretable than tree-based models
* Training time increases with dataset size

---

## Application in This Project

The machine learning pipeline implemented in this project follows these steps:

```text
SECOM Dataset
      ↓
Missing Value Handling
      ↓
Feature Scaling
      ↓
SMOTE Oversampling
      ↓
PCA / Feature Selection
      ↓
SVM Training
      ↓
Hyperparameter Optimization
      ↓
Performance Evaluation
```

The objective is to identify semiconductor manufacturing failures by learning an optimal decision boundary that maximizes the separation between pass and fail instances.

---




## Dataset

**Dataset:** SECOM Manufacturing Process Dataset

The dataset contains sensor measurements collected from a semiconductor manufacturing process. The target variable indicates whether a manufacturing run passed or failed quality control.

### Target Labels

| Label | Meaning |
| ----- | ------- |
| -1    | Pass    |
| 1     | Fail    |

### Dataset Characteristics

* High-dimensional sensor data
* Significant missing values
* Highly imbalanced target classes
* Suitable for anomaly and failure prediction tasks

---

## Project Workflow

### Phase 1: Data Ingestion & Exploratory Data Analysis

The dataset is downloaded using KaggleHub and loaded into a Pandas DataFrame.

Key preprocessing steps:

* Remove timestamp column (`Time`)
* Rename target column (`Pass/Fail` → `Target`)
* Examine class distribution
* Inspect dataset structure

---

### Phase 2: Data Preprocessing

To prepare the dataset for machine learning:

#### 1. Missing Value Handling

* Remove features with more than 50% missing values
* Impute remaining missing values using median imputation

#### 2. Feature Cleaning

* Remove constant (zero-variance) features

#### 3. Train-Test Split

* Stratified split to preserve class distribution
* 80% training, 20% testing

#### 4. Feature Scaling

* StandardScaler applied to all features

#### 5. Class Imbalance Handling

* SMOTE (Synthetic Minority Oversampling Technique)
* Applied only on training data to avoid data leakage

---

### Phase 3: Dimensionality Reduction

Two approaches are compared.

#### Approach A: PCA

Principal Component Analysis is used to retain 95% of the total variance while reducing dimensionality.

Benefits:

* Eliminates multicollinearity
* Compresses high-dimensional data
* Preserves most information

#### Approach B: Feature Selection

A Random Forest classifier computes feature importance scores.

* Top 50 most important features are retained
* Maintains interpretability
* Removes less informative sensors

---

### Phase 4: Model Training

A Support Vector Machine (SVM) with an RBF kernel is trained on both reduced datasets.

#### Hyperparameter Tuning

GridSearchCV is used with 3-fold cross-validation.

Parameter grid:

```python
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf']
}
```

Optimization metric:

* Recall Score

Recall is prioritized because detecting failures is more important than minimizing false positives.

---

### Phase 5: Evaluation

Each model is evaluated using:

#### Classification Metrics

* Precision
* Recall
* F1-Score
* Accuracy

#### Confusion Matrix

Visual representation of:

* True Positives
* False Positives
* True Negatives
* False Negatives

#### ROC-AUC Score

Measures overall classification performance across decision thresholds.

#### ROC Curve

Compares true positive rate against false positive rate.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn (SMOTE)
* KaggleHub

---

## Project Structure

```text
├── Semiconductor_Failure_Prediction.ipynb
├── README.md
├── requirements.txt
└── results/
    ├── confusion_matrix_pca.png
    ├── confusion_matrix_fs.png
    ├── roc_curve_pca.png
    └── roc_curve_fs.png
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/semiconductor-failure-prediction.git
cd semiconductor-failure-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn kagglehub
```

---

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Semiconductor_Failure_Prediction.ipynb
```

Run all cells sequentially.

---

## Results

The project compares two dimensionality reduction strategies before SVM classification:

| Approach | Dimensionality Reduction                          | Classifier |
| -------- | ------------------------------------------------- | ---------- |
| Model A  | PCA (95% Variance Retained)                       | SVM        |
| Model B  | Random Forest Feature Selection (Top 50 Features) | SVM        |

Performance comparison is based on:

* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

The best-performing approach can be selected based on the business objective of maximizing failure detection.

---

## Future Improvements

* Recursive Feature Elimination (RFE)
* XGBoost and LightGBM comparison
* Deep Learning approaches
* AutoML experimentation
* SHAP-based feature interpretation
* Advanced imbalance handling techniques

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow for semiconductor failure prediction. By comparing PCA-based dimensionality reduction with Random Forest feature selection, it evaluates the trade-off between feature compression and feature importance preservation before SVM classification. The pipeline provides a practical framework for predictive maintenance and quality control applications in semiconductor manufacturing.
