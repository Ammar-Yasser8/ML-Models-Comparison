import os

datasets = {
    "iris": "from sklearn.datasets import load_iris\nX, y = load_iris(return_X_y=True)\n",
    "breast_cancer": "from sklearn.datasets import load_breast_cancer\nX, y = load_breast_cancer(return_X_y=True)\n",
    "titanic": """import seaborn as sns\nimport pandas as pd\ndf = sns.load_dataset('titanic')\ndf.dropna(subset=['survived', 'age', 'fare'], inplace=True)\nX = pd.get_dummies(df[['pclass', 'sex', 'age', 'fare']], drop_first=True)\ny = df['survived']\n"""
}

models = {
    "knn": "from sklearn.neighbors import KNeighborsClassifier\nmodel = KNeighborsClassifier()\n",
    "lr": "from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(max_iter=2000)\n",
    "dt": "from sklearn.tree import DecisionTreeClassifier\nmodel = DecisionTreeClassifier()\n",
    "svm": "from sklearn.svm import SVC\nmodel = SVC()\n",
    "rf": "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier()\n"
}

base_code = """from pathlib import Path
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

sys.path.append(str(Path(__file__).resolve().parent.parent))
from report_utils import save_model_report

{dataset_code}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

{model_code}

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

save_model_report(
    dataset_name="{dataset_name}",
    model_name="{model_name}",
    accuracy=accuracy,
    precision=precision,
    recall=recall,
    f1_score=f1,
    confusion=matrix,
    class_report=report,
    output_path=Path(__file__).resolve().parent / "reports" / "{model_name}_report.html",
)
"""

for ds_name, ds_code in datasets.items():
    os.makedirs(ds_name, exist_ok=True)
    for mod_name, mod_code in models.items():
        code = base_code.format(dataset_code=ds_code, model_code=mod_code, dataset_name=ds_name, model_name=mod_name)
        with open(f"{ds_name}/{mod_name}.py", "w") as f:
            f.write(code)

print("Files created successfully.")