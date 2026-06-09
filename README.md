# Semiconductor Failure Prediction using SVM

## Overview

This project implements a complete machine learning pipeline for predicting semiconductor manufacturing failures using the SECOM dataset from the UCI Machine Learning Repository. The primary objective is to identify defective semiconductor production instances using Support Vector Machines (SVM) while comparing two dimensionality reduction approaches:

1. **Principal Component Analysis (PCA)**
2. **Feature Selection using Random Forest Feature Importance**

The project demonstrates data preprocessing, handling missing values, class imbalance treatment, dimensionality reduction, hyperparameter optimization, and model evaluation.

---


# Theory of Support Vector Machine (SVM)

## Introduction

Support Vector Machine (SVM) is a supervised machine learning algorithm used for both classification and regression tasks. It is most commonly used for classification problems and is particularly effective when dealing with high-dimensional data.

The main objective of SVM is to find the optimal decision boundary that separates different classes while maximizing the distance between the boundary and the nearest data points from each class.

In this project, SVM is used to classify semiconductor manufacturing instances as either:

* **Pass (-1)**
* **Fail (1)**

---

## Hyperplane

A hyperplane is the decision boundary that separates classes in the feature space.

For a two-dimensional dataset, the hyperplane can be represented as:

```text
w1x1 + w2x2 + b = 0
```

More generally:

```text
wᵀx + b = 0
```

where:

* `w` = weight vector
* `x` = feature vector
* `b` = bias term

The hyperplane divides the feature space into regions corresponding to different classes.

---

## Margin

The margin is the distance between the hyperplane and the nearest data points from each class.

SVM attempts to maximize this margin:

```text
Margin = 2 / ||w||
```

A larger margin generally results in:

* Better generalization
* Reduced overfitting
* Improved robustness to noise

---

## Support Vectors

Support vectors are the data points that lie closest to the hyperplane.

These points are important because:

* They determine the position of the decision boundary.
* They define the margin.
* They have the greatest influence on the classification model.

Unlike many machine learning algorithms, SVM relies primarily on these critical points rather than the entire dataset.

---

## Optimization Objective

For linearly separable data, SVM aims to:

```text
Minimize: (1/2) ||w||²
```

Subject to:

```text
yi(wᵀxi + b) ≥ 1
```

where:

* `xi` = feature vector
* `yi` = class label (+1 or -1)

Minimizing `||w||²` effectively maximizes the margin between classes.

---

## Soft Margin SVM

Real-world datasets often contain noise and overlapping classes, making perfect separation impossible.

To handle such situations, SVM introduces slack variables and a regularization parameter `C`.

The optimization problem becomes:

```text
Minimize: (1/2) ||w||² + C Σξi
```

where:

* `C` = regularization parameter
* `ξi` = penalty for misclassified points

### Effect of C

#### Large C

* Strong penalty for misclassification
* Smaller margin
* Lower training error
* Higher risk of overfitting

#### Small C

* Allows more classification errors
* Larger margin
* Better generalization
* Lower risk of overfitting

In this project, multiple values of `C` are tested using GridSearchCV.

---

## Non-Linear Classification

Many datasets cannot be separated using a straight line or linear hyperplane.

For example, if one class surrounds another, no linear boundary can perfectly separate them.

To solve this problem, SVM uses the **Kernel Trick**.

---

## Kernel Trick

The kernel trick allows SVM to transform data into a higher-dimensional space where classes become separable.

Instead of explicitly performing the transformation, SVM computes similarities between data points using kernel functions.

---

## Common Kernel Functions

### Linear Kernel

Used when the data is approximately linearly separable.

```text
K(xi, xj) = xiᵀxj
```

---

### Polynomial Kernel

Captures polynomial relationships between features.

```text
K(xi, xj) = (xiᵀxj + c)^d
```

where:

* `c` = constant term
* `d` = polynomial degree

---

### Radial Basis Function (RBF) Kernel

The most commonly used kernel.

```text
K(xi, xj) = exp(-γ ||xi - xj||²)
```

where:

* `γ` (gamma) controls the influence of training examples.

The RBF kernel can model highly complex nonlinear decision boundaries and is used in this project.

---

## Gamma Parameter

Gamma (`γ`) determines how far the influence of a training example extends.

### Small Gamma

* Smooth decision boundary
* Considers distant points
* Lower variance
* Better generalization

### Large Gamma

* Complex decision boundary
* Focuses on nearby points
* Higher variance
* Increased risk of overfitting

This project evaluates multiple gamma values during hyperparameter tuning.

---

## Importance of Feature Scaling

SVM relies on distance calculations, making feature scaling essential.

Consider two features:

| Feature     | Range    |
| ----------- | -------- |
| Temperature | 0 – 1000 |
| Pressure    | 0 – 1    |

Without scaling, the temperature feature would dominate the distance calculations.

To prevent this issue, features are standardized using:

```text
z = (x - μ) / σ
```

where:

* `μ` = mean of the feature
* `σ` = standard deviation of the feature

In this project, StandardScaler is used to normalize all sensor measurements before training.

---

## Why SVM is Suitable for Semiconductor Failure Prediction

The SECOM dataset presents several challenges:

* High-dimensional sensor data
* Nonlinear relationships between variables
* Class imbalance
* Limited failure samples

SVM is well-suited for these characteristics because it:

* Performs effectively in high-dimensional spaces
* Handles nonlinear patterns using kernels
* Provides strong generalization performance
* Works well when the number of features is large

---

## Advantages of SVM

* Effective in high-dimensional datasets
* Strong theoretical foundation
* Good generalization capability
* Supports nonlinear classification through kernels
* Uses only support vectors to define decision boundaries

---

## Limitations of SVM

* Computationally expensive for large datasets
* Sensitive to hyperparameter tuning
* Requires feature scaling
* Less interpretable than decision trees
* Training time increases with dataset size

---

## Application in This Project

The workflow used in this semiconductor failure prediction project is:

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
Hyperparameter Tuning
      ↓
Model Evaluation
```

The objective is to learn an optimal decision boundary that can accurately distinguish between successful and failed semiconductor manufacturing instances.

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
