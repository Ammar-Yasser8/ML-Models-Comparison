from pathlib import Path
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

sys.path.append(str(Path(__file__).resolve().parent.parent))
from report_utils import save_model_report

import seaborn as sns
import pandas as pd
df = sns.load_dataset('titanic')
df.dropna(subset=['survived', 'age', 'fare'], inplace=True)
X = pd.get_dummies(df[['pclass', 'sex', 'age', 'fare']], drop_first=True)
y = df['survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC
model = SVC()


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

save_model_report(
    dataset_name="titanic",
    model_name="svm",
    accuracy=accuracy,
    precision=precision,
    recall=recall,
    f1_score=f1,
    confusion=matrix,
    class_report=report,
    output_path=Path(__file__).resolve().parent / "reports" / "svm_report.html",
)
