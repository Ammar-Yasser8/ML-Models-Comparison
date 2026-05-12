# ML Models Comparison

This project compares classic machine learning classifiers across three datasets:

- Iris
- Breast Cancer
- Titanic

For each dataset, the repository includes scripts for:

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

## What Each Script Does

Each model script follows this flow:

1. Load a dataset.
2. Split into train and test sets using an 80/20 split.
3. Train one model.
4. Predict on the test set.
5. Print accuracy.
6. In some datasets, generate an HTML report.

## Datasets And Preprocessing

### Iris

- Source: `sklearn.datasets.load_iris()`
- Features: sepal length, sepal width, petal length, petal width
- Target: iris class label
- Preprocessing: none

### Breast Cancer

- Source: `sklearn.datasets.load_breast_cancer()`
- Features: tumor-related numeric features from sklearn dataset
- Target: malignant or benign class
- Preprocessing: none

### Titanic

- Source: `seaborn.load_dataset('titanic')`
- Features selected: `pclass`, `sex`, `age`, `fare`
- Target: `survived`
- Preprocessing:
  - Drops rows with missing values in `survived`, `age`, and `fare`
  - Encodes categorical columns with `pandas.get_dummies(..., drop_first=True)`

## Models And Main Parameters

- KNN: `KNeighborsClassifier()`
- Logistic Regression: `LogisticRegression(max_iter=2000)`
- Decision Tree:
  - Breast Cancer and Iris variants use `random_state=42`
  - Titanic variant uses default settings
- Random Forest:
  - Breast Cancer and Iris variants use `random_state=42`
  - Titanic variant uses default settings
- SVM: `SVC()`

## Reports

### Breast Cancer

All five scripts generate HTML reports under `breast_cancer/reports/`.

Each report includes:

- Accuracy metric
- Test predictions table
- Full dataset table

### Iris

HTML reports are generated under `iris/reports/`.

Current scripts in this repository generate reports for all five models listed in the folder.

### Titanic

Titanic scripts currently print accuracy to the console and do not generate HTML reports.

## Requirements

Install required packages:

```bash
pip install scikit-learn pandas seaborn
```

## How To Run

From repository root:

### Run One Script

```bash
python breast_cancer/dt.py
python iris/knn.py
python titanic/lr.py
```

### Run All Scripts (PowerShell)

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

`setup_datasets_models.py` is a generator that creates baseline dataset/model scripts.

It is useful for quickly recreating template scripts, then customizing them as needed.

## Notes

- The codebase currently mixes two report styles:
  - shared utility in `breast_cancer/report_utils.py`
  - inline HTML generation in some Iris scripts
- Reproducibility is strongest where `random_state=42` is used.

## License

No explicit license file is included in this repository yet.
