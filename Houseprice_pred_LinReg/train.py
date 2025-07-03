import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

# Load data
df = pd.read_csv("Hyderbad_House_price.csv")

df['BHK'] = df['title'].str.extract(r'(\d+)').astype(float)

df = df.dropna(subset=['BHK', 'area_insqft', 'location', 'price(L)'])

df = df[(df['area_insqft'] > 300) & (df['price(L)'] > 10)]

X = df[['area_insqft', 'location', 'BHK']]
y = df['price(L)']


# Preprocessing: One-Hot Encode 'location'

preprocessor = ColumnTransformer([
    ('location_encoder', OneHotEncoder(handle_unknown='ignore'), ['location'])
], remainder='passthrough')


# Pipeline: Preprocessing + Linear Regression

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', LinearRegression())
])


# Train-test split & train

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# Evaluate model

score_train = pipeline.score(X_train, y_train)
score_test = pipeline.score(X_test, y_test)

print(f"Train R² score: {score_train:.3f}")
print(f"Test R² score: {score_test:.3f}")


# Save trained model

os.makedirs("model", exist_ok=True)

with open("model/linear_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved to model/linear_model.pkl")
