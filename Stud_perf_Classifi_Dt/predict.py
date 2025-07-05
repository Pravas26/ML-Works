import joblib
import numpy as np

# Load trained model
model = joblib.load("models/decision_tree.pkl")

print("\nStudent Performance Predictor")
print("Please enter the following information:")

# User input
try:
    hours = float(input("Hours Studied: "))
    attendance = float(input("Attendance (%): "))
    assignments = int(input("Assignments Completed (1 = Yes, 0 = No): "))
    
    features = np.array([[hours, attendance, assignments]])
    prediction = model.predict(features)[0]

    print("\nPrediction:", "Pass" if prediction == 1 else "Fail")
except Exception as e:
    print("Error:", e)
