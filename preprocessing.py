# ==========================
# Step 1: Import Libraries
# ==========================
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================
# Step 2: Load Dataset
# ==========================
df = pd.read_csv("student_dataset_10000_rows (1).csv")

# ==========================
# Step 3: Display Dataset
# ==========================
print(df.head())
print(df.info())
print(df.describe())

# ==========================
# Step 4: Check Missing Values
# ==========================
print("\nMissing Values:")
print(df.isnull().sum())

# ==========================
# Step 5: Remove Duplicate Rows
# ==========================
df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)

# ==========================
# Step 6: Encode Categorical Column
# ==========================
label_encoder = LabelEncoder()

df["placement_status"] = label_encoder.fit_transform(df["placement_status"])

# ==========================
# Step 7: Separate Features and Target
# ==========================
X = df.drop("placement_status", axis=1)
y = df["placement_status"]

# ==========================
# Step 8: Feature Scaling
# ==========================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X = pd.DataFrame(X_scaled, columns=X.columns)

# ==========================
# Step 9: Display Processed Data
# ==========================
print("\nProcessed Features:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nPreprocessing Completed Successfully!")