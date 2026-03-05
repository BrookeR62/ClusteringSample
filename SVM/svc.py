# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# ==============================
# 2. LOAD CSV DATASET
# ==============================
iris_df = pd.read_csv("SVM/winequality-red.csv")

print("=== DATA INFO ===")
print(iris_df.info())

print("\n=== DATA DESCRIPTION ===")
print(iris_df.describe())

print("\n=== DUPLICATE ROWS ===")
print(iris_df.duplicated().sum())

# ==============================
# 3. DATA CLEANING
# ==============================

# Drop duplicates
iris_df = iris_df.drop_duplicates()
iris_df.reset_index(drop=True, inplace=True)

# Handle missing values (impute numeric with mean)
for col in iris_df.select_dtypes(include=np.number).columns:
    iris_df[col].fillna(iris_df[col].mean(), inplace=True)

# ==============================
# 4. OUTLIER REMOVAL (IQR METHOD)
# ==============================

def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

numeric_cols = iris_df.select_dtypes(include=np.number).columns
iris_df = remove_outliers_iqr(iris_df, numeric_cols)

# ==============================
# 5. EXPLORATORY DATA ANALYSIS
# ==============================

# Histograms
iris_df.hist(figsize=(10,8))
plt.suptitle("Feature Distributions")
plt.show()

# Boxplots per species
plt.figure(figsize=(12,8))
for i, col in enumerate(iris_df.columns[:-1]):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=col, data=iris_df)
    plt.title(f"{col} by species")
plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(iris_df, hue='species', palette='Set2')
plt.suptitle("Pairplot of Features Colored by Species", y=1.02)
plt.show()

# ==============================
# 6. FEATURE / TARGET SPLIT
# ==============================

X = iris_df.drop("species", axis=1)
y = iris_df["species"]

# ==============================
# 7. TRAIN TEST SPLIT (80/20)
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=52,
    stratify=y
)

# ==============================
# 8. STANDARDIZATION
# ==============================

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ==============================
# 9. HYPERPARAMETER EXPERIMENT
# ==============================

kernels = ["linear", "poly", "rbf", "sigmoid"]
c_values = [0.1, 1, 10]
gamma_values = [0.1, 0.5, 1]

results = []

for kernel in kernels:
    for C in c_values:
        for gamma in gamma_values:
            try:
                model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
                model.fit(X_train_sc, y_train)
                y_pred = model.predict(X_test_sc)

                results.append({
                    "Kernel": kernel,
                    "C": C,
                    "Gamma": gamma,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision_macro": precision_score(
                        y_test, y_pred,
                        average="macro",
                        zero_division=0
                    )
                })
            except:
                continue

df_results = pd.DataFrame(results)

print("\n=== HYPERPARAMETER PERFORMANCE TABLE ===")
print(df_results.sort_values(by="Accuracy", ascending=False).round(3))

# Pivot table format
pivot_table = df_results.pivot_table(
    index=["Kernel", "C"],
    columns="Gamma",
    values="Accuracy"
).round(3)

print("\n=== Pivot Table (Accuracy) ===")
print(pivot_table)

# ==============================
# 10. BEST MODEL SELECTION
# ==============================

best_row = df_results.sort_values(by="Accuracy", ascending=False).iloc[0]

best_model = SVC(
    kernel=best_row["Kernel"],
    C=best_row["C"],
    gamma=best_row["Gamma"],
    random_state=42
)

best_model.fit(X_train_sc, y_train)
best_pred = best_model.predict(X_test_sc)

print("\n=== BEST MODEL PARAMETERS ===")
print(best_row)

print("\nAccuracy:", accuracy_score(y_test, best_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, best_pred))

# ==============================
# 11. CONFUSION MATRIX
# ==============================

labels = sorted(y.unique())
cm = confusion_matrix(y_test, best_pred, labels=labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Best SVC Model")
plt.show()

# ==============================
# 12. ARBITRARY PREDICTION
# ==============================

new_sample = [[5.1, 3.5, 1.4, 0.2]]
new_sample_scaled = scaler.transform(new_sample)

prediction = best_model.predict(new_sample_scaled)

print("\nPrediction for new sample:", prediction[0])