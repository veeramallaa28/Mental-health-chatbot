import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

print("Starting model training (Scale 1-5)...")

# 1. Load your dataset from the CSV file
try:
    df = pd.read_csv('risk_data.csv')
except FileNotFoundError:
    print("Error: risk_data.csv not found. Please make sure it's in the same folder.")
    exit()

X = df['text']
y = df['label']

# 3.  model pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# 4. Train the model on the entire dataset
pipeline.fit(X, y)

# 5. Save the trained pipeline to a file
joblib.dump(pipeline, 'risk_classifier.joblib')

print("Model training complete. Model saved to 'risk_classifier.joblib'")