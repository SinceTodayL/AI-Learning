## Learning What in SITP Project: Research on big data analysis framework for Maglev track anomaly detection



主要是用Python处理大数据的一些方法，涉及到机器学习的比较多



### KNN (k-nearest-neighbors classification)

理解起来最简单的一种算法: 就是根据最近邻居的标签来判断，步骤就是先计算与样本中每个点之间的距离，然后排序，根据最近的k个点的标签进行投票，投票结果就是预测结果

注意在处理数据之前需要 normalize (将均值变为0，方差变为1)，因为该算法需要计算点之间的距离，尺度统一化很重要（虽然其实在本数据库中影响不大）

```Python
import pandas as pd

# sklearn 是一个很重要的机器学习库，里面有很多模块
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# r表示原始字符串，避免反斜杠被转义
data_path = r".\SITP Project Practice\Data.xlsx"
label_path = r".\SITP Project Practice\DataLabel.xlsx"

'''
  pandas库里面的 read_excel 函数，返回的是一个Dataframe
  默认的效果是会把表格的第一行当作标签名称，比如这个Data数据集
  如果第一行不是标签，则需要加上 header=None
  Dataframe相比于普通二维数组的优势是可以根据标签来访问列元素
'''

data = pd.read_excel(data_path)
label = pd.read_excel(label_path)

'''
  .values 可以将数据转化为ndarray的形式，即numpy数组
  numpy数组是用 C语言编写的底层代码，性能优于列表
  同时也支持大量数学运算，下面的标准化操作也需要用到的是numpy数组
  .iloc可以访问某个元素，如 data.iloc[2, 3]
  也可以如下所示访问某一块元素，如果是 : 就代表从第一行一直到最后一行
'''
X = data.iloc[ : ,  : ].values
y = label.iloc[ : ].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 这里 random_state 的设置是为了让每一次的划分结果一致，便于实验重现
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 30)

'''
  KNN 核心代码
'''
k = 15
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy : {accuracy * 100:.2f}%")


'''
  matplotlib也是一个重要的库
'''
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.xlabel("Pred")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
```

