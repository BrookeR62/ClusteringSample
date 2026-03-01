import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

df = pd.read_csv('SVM/irisdatasets.csv')

le = LabelEncoder()
df['Species_num'] = le.fit_transform(df['species'])  # setosa=0, versicolor=1, virginica=2


# Sa features ug target
X = df.iloc[:, 0:4]           
y = df['Species_num']         

#split sa data pang training ug testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train ug evaluate sa SVR models with different kernels and C values
kernels = ["linear", "rbf", "poly", "sigmoid"]
C_values = [0.1, 1, 10]

results = []

for kernel in kernels:
    mse_results = []
    print(f"\nTraining Kernel: {kernel}")
    
    for C in C_values:
        model = SVR(kernel=kernel, C=C)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mse_results.append(mse)
        print(f"  C={C} -> MSE={mse:.4f}")
    
    results.append([kernel] + mse_results)



#pang display sa results
results_df = pd.DataFrame(results, columns=[
    "Kernel",
    "C1 (C=0.1) MSE",
    "C2 (C=1) MSE",
    "C3 (C=10) MSE"
])

print("\n==============================================")
print("SVR PERFORMANCE ON IRIS DATASET")
print("==============================================")
print(results_df.to_string(index=False))