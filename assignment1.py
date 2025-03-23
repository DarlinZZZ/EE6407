import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from scipy.stats import zscore

# ===== Step 1: Load and preprocess training data =====
train_file = "TrainingData.xlsx"
df_raw = pd.read_excel(train_file, header=None)

# Check if first row is header
first_row = df_raw.iloc[0]
if all(col in ['A', 'B', 'C', 'D', 'Label'] for col in first_row):
    df_raw = df_raw.iloc[1:]

df_raw.columns = ['A', 'B', 'C', 'D', 'Label']
df_raw = df_raw.reset_index(drop=True)

# Replace '?' with NaN
df_nan = df_raw.replace('?', np.nan)

# ===== Missing value removal =====
missing_rows = df_nan[df_nan.isnull().any(axis=1)].copy()
missing_rows['Original_Excel_Row'] = missing_rows.index + 2
print("\n--- Removed Rows with Missing Values ---")
print(missing_rows[['A', 'B', 'C', 'D', 'Label', 'Original_Excel_Row']])

df_no_missing = df_nan.dropna().astype(float)

# ===== Outlier removal using Z-score =====
features = df_no_missing[['A', 'B', 'C', 'D']]
z_scores = zscore(features)
non_outliers = (np.abs(z_scores) <= 3).all(axis=1)
outlier_rows = df_no_missing[~non_outliers].copy()
outlier_rows['Original_Excel_Row'] = outlier_rows.index + 2 + len(missing_rows)

print("\n--- Removed Rows with Outliers (Z-score > 3) ---")
print(outlier_rows[['A', 'B', 'C', 'D', 'Label', 'Original_Excel_Row']])

df_cleaned = df_no_missing[non_outliers]

# Save cleaned training set (no header)
df_cleaned.to_excel("trainingprune.xlsx", index=False, header=False)

# ===== Step 2: Train Naive Bayes Classifier =====
X_train = df_cleaned[['A', 'B', 'C', 'D']]
y_train = df_cleaned['Label']
model = GaussianNB()
model.fit(X_train, y_train)

print("\n=== Naive Bayes Model Trained ===")
print("Class Priors:", model.class_prior_)
print("Class Means:\n", model.theta_)
print("Class Variances:\n", model.var_)

# ===== Step 2.5: Print Full Bayes Classifier Parameters =====
print("\n=== Bayes Decision Rule Classifier Parameters ===")
classes = np.unique(y_train)

for c in classes:
    X_c = X_train[y_train == c]
    mu_c = X_c.mean().values
    sigma_c = np.cov(X_c.T)

    print(f"\nFor class {int(c)}, the mean vector μ{int(c)} is:")
    print(mu_c.round(4).tolist())

    print(f"\nΣ{int(c)} =")
    print(np.round(sigma_c, 4))

# ===== Step 2.6: Print Naive Bayes Parameters =====
print("\n=== Naive Bayes Classifier Parameters ===")
for idx, c in enumerate(model.classes_):
    print(f"\nFor class {int(c)}, the mean vector μ{int(c)} is:")
    print(np.round(model.theta_[idx], 4).tolist())

    print(f"\nσ²{int(c)} =")
    print(np.round(model.var_[idx], 4).tolist())

# ===== Step 3: Load and Predict Test Data =====
test_file = "TestData.xlsx"
df_test = pd.read_excel(test_file, header=None)
df_test.columns = ['A', 'B', 'C', 'D']
df_test = df_test.astype(float)

# Predict
predictions = model.predict(df_test)

# Save predictions to Excel (no header)
pd.DataFrame(predictions).to_excel("PredictedLabels.xlsx", index=False, header=False)
print("\n✅ Predictions saved to 'PredictedLabels.xlsx'")

