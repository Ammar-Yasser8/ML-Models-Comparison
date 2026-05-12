from pathlib import Path
from html import escape

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


dataset = load_iris()
X = dataset.data
y = dataset.target
feature_names = dataset.feature_names
target_names = dataset.target_names

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")


def build_dataset_table() -> str:
	header_cells = "".join(f"<th>{escape(name)}</th>" for name in feature_names)
	header_cells += "<th>target</th><th>target_name</th>"

	rows = []
	for features, label in zip(X, y):
		row_cells = "".join(f"<td>{value:.2f}</td>" for value in features)
		row_cells += f"<td>{int(label)}</td><td>{escape(target_names[int(label)])}</td>"
		rows.append(f"<tr>{row_cells}</tr>")

	return (
		"<table><thead><tr>"
		+ header_cells
		+ "</tr></thead><tbody>"
		+ "".join(rows)
		+ "</tbody></table>"
	)


def build_test_predictions_table() -> str:
	header_cells = "".join(f"<th>{escape(name)}</th>" for name in feature_names)
	header_cells += "<th>actual</th><th>predicted</th><th>match</th>"

	rows = []
	for features, actual, pred in zip(X_test, y_test, y_pred):
		row_cells = "".join(f"<td>{value:.2f}</td>" for value in features)
		row_cells += (
			f"<td>{escape(target_names[int(actual)])}</td>"
			f"<td>{escape(target_names[int(pred)])}</td>"
			f"<td>{'yes' if int(actual) == int(pred) else 'no'}</td>"
		)
		rows.append(f"<tr>{row_cells}</tr>")

	return (
		"<table><thead><tr>"
		+ header_cells
		+ "</tr></thead><tbody>"
		+ "".join(rows)
		+ "</tbody></table>"
	)


html = f"""<!doctype html>
<html lang=\"en\">
<head>
	<meta charset=\"utf-8\" />
	<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
	<title>Iris KNN Report</title>
	<style>
		body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #1f2937; background: #f8fafc; }}
		h1, h2 {{ margin: 0 0 12px; }}
		p {{ margin: 0 0 16px; }}
		.panel {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 16px; }}
		.metric {{ font-size: 1.1rem; font-weight: 700; color: #1d4ed8; }}
		.table-wrap {{ overflow-x: auto; }}
		table {{ border-collapse: collapse; width: 100%; min-width: 800px; }}
		th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; }}
		th {{ background: #e2e8f0; position: sticky; top: 0; }}
		tr:nth-child(even) {{ background: #f8fafc; }}
	</style>
</head>
<body>
	<h1>Iris - KNN Report</h1>
	<div class=\"panel\">
		<p class=\"metric\">Accuracy: {accuracy:.4f}</p>
		<p>Rows in full dataset: {len(X)} | Rows in test set: {len(X_test)}</p>
	</div>

	<div class=\"panel\">
		<h2>Test Predictions</h2>
		<div class=\"table-wrap\">{build_test_predictions_table()}</div>
	</div>

	<div class=\"panel\">
		<h2>Full Dataset (All Rows)</h2>
		<div class=\"table-wrap\">{build_dataset_table()}</div>
	</div>
</body>
</html>
"""

reports_dir = Path(__file__).parent / "reports"
reports_dir.mkdir(exist_ok=True)
report_path = reports_dir / "knn_report.html"
report_path.write_text(html, encoding="utf-8")

print(f"Report generated: {report_path}")
