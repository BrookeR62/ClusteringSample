import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, confusion_matrix,  ConfusionMatrixDisplay


# Load data
iris_df = pd.read_csv('SVM/irisdatasets.csv')
le = LabelEncoder()
iris_df['species'] = le.fit_transform(iris_df['species'])

X = iris_df.drop("species", axis=1)
y = iris_df["species"]

#split the data set into 20% testing and 80% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42, stratify=y)

#feature scaling
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#train SVM in different kernels and c value
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
c_value = [0.1, 1, 10, 100]

results = {}

for kernel in kernels: 
    results[kernel] = []
    for C in c_value:
        model = SVC(kernel=kernel, C=C)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        precision = precision_score(y_test, y_pred, average='macro')
        results[kernel].append(round(precision, 4))

result_df = pd.DataFrame(results, index = ['C1', 'C2', 'C3']).T 
print(result_df)
print(classification_report(y_test, y_pred, target_names=le.classes_))