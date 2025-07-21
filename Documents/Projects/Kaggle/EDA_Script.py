import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, classification_report
import Kaggle_Extractor  # Your custom module (must be in the same folder)

dataset = "sebastianwillmann/beverage-sales"  # Change to any Kaggle dataset you want
all_data = Kaggle_Extractor.download_and_load_all_csvs(dataset)

print("\n✅ Available files:", list(all_data.keys()))

df = all_data["synthetic_beverage_sales_data.csv"]
print(df.shape)
print(df.columns)
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())  # Check missing values

numeric_data = df.select_dtypes(include=[np.number])
means = np.mean(numeric_data)
print("Column Means:\n", means)

# Histogram of all numeric features
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------------------
# Step 5: Example ML Model (Linear Regression)
# -----------------------------------------
# Drop missing rows (simple handling for demo)
df = df.dropna()

# Choose a target column (CHANGE THIS to a real numeric column in your data)
target_col = "your_target_column"  # 🔁 Replace with actual column name
if target_col not in df.columns:
    print(f"❌ ERROR: Column '{target_col}' not found in dataset.")
else:
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Evaluation
    print("\n📈 Linear Regression Performance:")
    print("MSE:", mean_squared_error(y_test, y_pred))

df.to_csv("kaggle_data/your_filename.csv", index=False)