import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("student_dataset_10000_rows (1).csv")

# ==========================
# Features and Target
# ==========================
X = df.drop("exam_score", axis=1)
y = df["exam_score"]

# ==========================
# Convert Categorical Column
# ==========================
X = pd.get_dummies(X, drop_first=True)

# ==========================
# Train-Test Split (85%-15%)
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42
)

# ==========================
# Feature Scaling
# ==========================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# Train Model
# ==========================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# Predictions
# ==========================
y_pred = model.predict(X_test)

# ==========================
# Accuracy
# ==========================
accuracy = r2_score(y_test, y_pred)

print(f"Model Accuracy (R² Score): {accuracy:.2f}")
# ==========================
# STEP 9: Save Model & Scaler
# ==========================

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("=" * 50)
print("🎉 Model trained successfully!")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("✅ model.pkl saved successfully.")
print("✅ scaler.pkl saved successfully.")
print("=" * 50)