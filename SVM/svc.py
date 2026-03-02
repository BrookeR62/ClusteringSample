import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, confusion_matrix
import matplotlib.pyplot as plt



df = pd.read_csv("SVM/irisdatasets.csv")

print("Dataset Preview:")
print(df.head(), "\n")



#Separate Features and Labels
X = df.drop("species", axis=1)
y = df["species"]


#split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


#Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




#Train SVM with Different Kernels and C Values
kernels = ["linear", "poly", "rbf", "sigmoid"]
C_values = [0.1, 1, 10]

results = []
models = {}

for kernel in kernels:
    for C in C_values:
        svm = SVC(kernel=kernel, C=C)
        svm.fit(X_train, y_train)

        y_pred = svm.predict(X_test)

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted"
        )

        results.append([kernel, C, precision])
        models[(kernel, C)] = svm


#display result
results_df = pd.DataFrame(
    results,
    columns=["Kernel", "C Value", "Precision"]
)

print("SVM Precision Results:\n")
print(results_df, "\n")


#mo pili ug nindot nga model
best_row = results_df.loc[results_df["Precision"].idxmax()]
best_kernel = best_row["Kernel"]
best_C = best_row["C Value"]

best_model = models[(best_kernel, best_C)]

print("Best SVM Model:")
print(f"Kernel: {best_kernel}")
print(f"C Value: {best_C}")
print(f"Precision: {best_row['Precision']}\n")


# confusion matrix for best model
y_best_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_best_pred)

print("Confusion Matrix:")
print(cm)
print("\nInterpretation:")
print("Rows represent ACTUAL classes")
print("Columns represent PREDICTED classes")
print("Diagonal values show correct classifications")