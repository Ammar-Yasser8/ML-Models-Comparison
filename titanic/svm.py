from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
