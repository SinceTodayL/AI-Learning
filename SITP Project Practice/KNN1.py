import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

data_path = r".\SITP Project Practice\Data.xlsx"
label_path = r".\SITP Project Practice\DataLabel.xlsx"

data = pd.read_excel(data_path)
label = pd.read_excel(label_path)

X = data.iloc[ : ,  : ].values
y = label.iloc[ : ].values.squeeze()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 这里 random_state 的设置是为了让每一次的划分结果一致，便于实验重现
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 30)

def KNN(k):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy : {accuracy * 100:.2f}%")

for k in range(2, 50, 2):
    KNN(k)

'''
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.xlabel("Pred")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
'''