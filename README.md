# ML Models Comparison

A compact benchmark project for comparing classic classification models on three datasets.

## Overview

This repository evaluates the following datasets:

- Iris
- Breast Cancer
- Titanic

For each dataset, the project includes:

- KNN
- Logistic Regression
- Decision Tree
- Random Forest
- SVM

## Project Structure

```text
AI-Tasks/
  setup_datasets_models.py
  breast_cancer/
    dt.py
    knn.py
    lr.py
    rf.py
    svm.py
    report_utils.py
    reports/
      dt_report.html
      knn_report.html
      lr_report.html
      rf_report.html
      svm_report.html
  iris/
    dt.py
    knn.py
    lr.py
    rf.py
    svm.py
    reports/
      dt_report.html
      knn_report.html
      lr_report.html
      rf_report.html
      svm_report.html
  titanic/
    dt.py
    knn.py
    lr.py
    rf.py
    svm.py
```

## Workflow

Each model script follows the same pipeline:

1. Load dataset
2. Split into train and test sets (80/20)
3. Train model
4. Predict test labels
5. Print accuracy
6. Optionally generate HTML report

## Dataset Details

| Dataset | Source | Features | Target | Preprocessing |
|---|---|---|---|---|
| Iris | sklearn.datasets.load_iris() | Sepal and petal measurements | Iris class | None |
| Breast Cancer | sklearn.datasets.load_breast_cancer() | Numeric tumor measurements | Malignant or benign | None |
| Titanic | seaborn.load_dataset('titanic') | pclass, sex, age, fare | survived | Drop missing values, one-hot encode categorical columns |

## Models And Parameters

| Model | Configuration |
|---|---|
| KNN | KNeighborsClassifier() |
| Logistic Regression | LogisticRegression(max_iter=2000) |
| Decision Tree | random_state=42 for Iris and Breast Cancer; default on Titanic |
| Random Forest | random_state=42 for Iris and Breast Cancer; default on Titanic |
| SVM | SVC() |

## Reports

- Breast Cancer: all five scripts generate HTML reports in breast_cancer/reports/.
- Iris: all five scripts generate HTML reports in iris/reports/.
- Titanic: scripts print accuracy to console only.

Typical report content:

- Accuracy
- Test predictions table
- Full dataset table

## Requirements

Install dependencies:

```bash
pip install scikit-learn pandas seaborn
```

## Usage

Run individual scripts from repository root:

```bash
python breast_cancer/dt.py
python iris/knn.py
python titanic/lr.py
```

Run all scripts in PowerShell:

```powershell
$datasets = @('breast_cancer', 'iris', 'titanic')
$models = @('dt', 'knn', 'lr', 'rf', 'svm')

foreach ($ds in $datasets) {
    foreach ($m in $models) {
        python "$ds/$m.py"
    }
}
```

## Generator Script

setup_datasets_models.py creates baseline model scripts for each dataset.

## Notes

- Reporting style differs across dataset folders.
- Results are most reproducible where random_state=42 is used.

## Copyright

Copyright (c) 2026 Ammar Yasser. All rights reserved.

