import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Split features and target
X = data.drop("label", axis=1)
y = data["label"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model properly
with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")