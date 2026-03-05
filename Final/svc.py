import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

iris_df = pd.read_csv("Final/irisdatasets.csv") 
print(iris_df.duplicated().sum()) 
iris_df = iris_df.drop_duplicates(subset=["sepal_length","sepal_width","petal_length","petal_width"])
print(iris_df.duplicated().sum()) 
iris_df.reset_index(drop=True, inplace=True) 
print(iris_df.head())



X = iris_df.drop("species", axis=1)
y = iris_df["species"]

iris_df.describe



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=52, stratify=y
)


scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)


kernels = ["linear", "poly", "rbf", "sigmoid"]
c_values = [0.1, 1, 10]
gamma_values = ["scale", "auto", 0.1, 0.5, 1]

results = []

for kernel in kernels:
    for C in c_values:
        for gamma in gamma_values:
            # Linear kernel ignores gamma, but we include it for table consistency
            try:
                model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
                model.fit(X_train_sc, y_train)
                y_pred = model.predict(X_test_sc)
                precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
                results.append({
                    "Kernel": kernel.capitalize(),
                    "C": C,
                    "Gamma": gamma,
                    "Precision": precision
                })
            except Exception as e:
                continue

df_results = pd.DataFrame(results)

result_table = df_results.pivot_table(
    index=["Kernel", "C"], columns="Gamma", values="Precision"
).round(3)

print("Performance Table (Precision by Kernel × C × Gamma):")
print(result_table)


best_row = df_results.sort_values(by="Precision", ascending=False).iloc[0]
best_kernel = best_row["Kernel"].lower()
best_C = best_row["C"]
best_gamma = best_row["Gamma"]

best_model = SVC(kernel=best_kernel, C=best_C, gamma=best_gamma, random_state=42)
best_model.fit(X_train_sc, y_train)
best_pred = best_model.predict(X_test_sc)

print("\nBest Model:")
print(f"Kernel: {best_kernel}, C: {best_C}, Gamma: {best_gamma}")
print("Accuracy:", accuracy_score(y_test, best_pred))
print("Precision (macro):", precision_score(y_test, best_pred, average="macro", zero_division=0))


labels = sorted(y.unique())
cm = confusion_matrix(y_test, best_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Best SVM")
plt.show()


new_sample = [[5.9, 3.0, 5.1, 1.8]]  
new_sample_sc = scaler.transform(new_sample)
predicted_species = best_model.predict(new_sample_sc)
print("\nPrediction for arbitrary input:", predicted_species[0])
model.fit