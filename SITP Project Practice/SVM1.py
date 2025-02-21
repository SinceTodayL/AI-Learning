import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# 生成线性可分数据
X, y = make_blobs(n_samples=500, centers=2, random_state=6)

# 训练SVM模型（线性核）
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

# 绘制数据点
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# 绘制超平面和间隔
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 生成网格点以绘制决策边界
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# 绘制超平面、间隔边界和支持向量
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')

# 添加标签和注释
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("SVM")
plt.show()