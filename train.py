# train.py

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import joblib
import os

# Load dataset
df = sns.load_dataset("penguins").dropna()

# One-hot encode categorical features
df = pd.get_dummies(df, columns=["sex", "island"])

# Label encode the target variable
label_encoder = LabelEncoder()
df["species"] = label_encoder.fit_transform(df["species"])

# Split data into features and target
X = df.drop("species", axis=1)
y = df["species"]

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train the model
model = XGBClassifier(max_depth=3, n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Evaluate the model
train_f1 = f1_score(y_train, model.predict(X_train), average='weighted')
test_f1 = f1_score(y_test, model.predict(X_test), average='weighted')

print(f"Train F1 Score: {train_f1}")
print(f"Test F1 Score: {test_f1}")

# Save the model, label encoder, and feature columns
os.makedirs("app/data", exist_ok=True)
joblib.dump((model, label_encoder, list(X.columns)), "app/data/model.json")
print("Model saved successfully.")
