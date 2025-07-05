import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib

# Sample DataFrame
data = {
    "Hours_Studied": [2, 4, 6, 8, 5, 9, 3, 7, 10, 1, 6, 5],
    "Attendance": [60, 75, 80, 85, 70, 90, 65, 88, 95, 50, 82, 76],
    "Assignments_Completed": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    "Pass": [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Features and target
X = df.drop("Pass", axis=1)
y = df["Pass"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))


os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/decision_tree.pkl")

# Visualize tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=X.columns, class_names=["Fail", "Pass"], filled=True)
plt.title("Decision Tree - Student Performance")
plt.tight_layout()
plt.savefig("models/tree.png")
print("Model saved to 'models/decision_tree.pkl' and tree saved to 'models/tree.png'")
