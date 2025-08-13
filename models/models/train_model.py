{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7487b6a6-6f9f-44f1-8a87-c92d4121771e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "",
   "name": ""
  },
  "language_info": {
   "name": ""
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv('../dataset/fake_job_postings.csv')

# Drop rows with missing target
df = df[df['fraudulent'].notna()]

# Combine relevant text columns into one (you can adjust)
df['text'] = df['title'].fillna('') + ' ' + df['location'].fillna('') + ' ' + df['description'].fillna('')

# Features and target
X = df['text']
y = df['fraudulent']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline: TF-IDF + Logistic Regression
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression())
])

# Train model
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
