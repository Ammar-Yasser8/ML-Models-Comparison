from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

from report_utils import generate_report


dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
feature_names = dataset.feature_names
target_names = dataset.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

report_path = generate_report(
	dataset_name="Breast Cancer",
	model_name="RF",
	report_filename="rf_report.html",
	X=X,
	y=y,
	X_test=X_test,
	y_test=y_test,
	y_pred=y_pred,
	feature_names=feature_names,
	target_names=target_names,
	accuracy=accuracy,
)
print(f"Report generated: {report_path}")
