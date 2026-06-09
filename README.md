# Semiconductor-Failure-Prediction-using-SVM

# Semiconductor Failure Prediction using SVM

## Overview

This project implements a complete machine learning pipeline for predicting semiconductor manufacturing failures using the SECOM dataset from the UCI Machine Learning Repository. The primary objective is to identify defective semiconductor production instances using Support Vector Machines (SVM) while comparing two dimensionality reduction approaches:

1. **Principal Component Analysis (PCA)**
2. **Feature Selection using Random Forest Feature Importance**

The project demonstrates data preprocessing, handling missing values, class imbalance treatment, dimensionality reduction, hyperparameter optimization, and model evaluation.

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
