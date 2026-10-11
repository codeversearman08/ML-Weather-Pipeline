import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification

# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================
# Note: The original Kaggle Daily Weather Dataset (2010–2024) is excluded from 
# this repository due to file size limits. A synthetic dataset generator is 
# included below to demonstrate the pipeline architecture and execution.

dataset_path = 'kaggle_daily_weather.csv'

if os.path.exists(dataset_path):
    print("Loading original Kaggle dataset...")
    df = pd.read_csv(dataset_path)
    features = ['Cloud_Cover', 'Humidity', 'Temperature', 'Pressure', 'Wind_Speed']
    X = df[features]
    y = df['Rain_Target']
else:
    print("Notice: Original dataset not found. Generating synthetic weather data to demonstrate pipeline execution...")
    X, y = make_classification(
        n_samples=1000, 
        n_features=5, 
        n_informative=4, 
        n_redundant=1, 
        random_state=42
    )

# Chronological/Standard Splitting (85% Train, 15% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# ==========================================
# 2. Model Initialization
# ==========================================
scaler = StandardScaler()

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_pipeline = Pipeline([('scaler', scaler), ('rf', rf_model)])

# XGBoost Model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_pipeline = Pipeline([('scaler', scaler), ('xgb', xgb_model)])

# Ensemble Model (Soft Voting)
ensemble_model = VotingClassifier(
    estimators=[('rf', rf_pipeline), ('xgb', xgb_pipeline)],
    voting='soft'
)

# ==========================================
# 3. Training & Evaluation 
# ==========================================
def evaluate_model(model, X, y, model_name="Model"):
    print(f"\n{'='*40}")
    print(f"--- {model_name} Evaluation ---")
    print(f"{'='*40}")
    
    predictions = model.predict(X)
    
    print(classification_report(y, predictions, target_names=['no rain', 'rain']))
    print(f"Accuracy: {accuracy_score(y, predictions):.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y, predictions))

# Execute Pipeline
if __name__ == "__main__":
    print("\nExecuting ML Weather Pipeline...")
    
    rf_pipeline.fit(X_train, y_train)
    evaluate_model(rf_pipeline, X_test, y_test, "Random Forest")
    
    xgb_pipeline.fit(X_train, y_train)
    evaluate_model(xgb_pipeline, X_test, y_test, "XGBoost")
    
    ensemble_model.fit(X_train, y_train)
    evaluate_model(ensemble_model, X_test, y_test, "Ensemble Model")
