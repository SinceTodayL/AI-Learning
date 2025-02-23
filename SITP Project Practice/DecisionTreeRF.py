import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

data_path = r".\SITP Project Practice\Data.xlsx"
label_path = r".\SITP Project Practice\DataLabel.xlsx"

data = pd.read_excel(data_path)
label = pd.read_excel(label_path)
X = data.values
y = label.values.squeeze()

class_name = ["Normal", "Mild", "Sereve"]

final_preds = np.zeros_like(y)

kf = KFold(n_splits=5, shuffle=True, random_state=41)
for flod_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)      
    # 相比上面而言缺少了 fit, 就会用上面训练集的标准化参数来处理测试集，而不重新对测试集计算方差、均值等

    '''
        First Layer : Normal vs Abnormal
    '''
    y_train_layer1 = np.where(y_train == 0, 0, 1)

    # using smote
    smote = SMOTE(random_state=41)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train_layer1)

    # train desicion tree model 
    model_layer1 = DecisionTreeClassifier(random_state=41, max_depth=10)  # 确定模型参数
    model_layer1.fit(X_train_res, y_train_res)                            # 传入训练集，训练模型          
    # predict
    y_pred_layer1 = model_layer1.predict(X_test_scaled)                   # 使用模型

    '''
        Second Layer : Mild vs Sereve
    '''
    abnormal_mask = (y_pred_layer1 == 1)
    X_test_abnormal = X_test_scaled[abnormal_mask]
    y_test_abnormal = y_test[abnormal_mask]

    if len(X_test_abnormal) > 0:
        abnormal_mask_train = (y_train != 0)
        X_train_abnormal = X_train_scaled[abnormal_mask_train]
        y_train_abnormal = y_train[abnormal_mask_train]
        y_train_layer2 = np.where(y_train_abnormal == 1, 0, 1)

        '''
            problem: 阈值确定？？
        '''
        # 对上一步已经预测为 abnormal 的数据再次进行 SMOTE
        X_train_abnormal_res, y_train_abnormal_res = smote.fit_resample(X_train_abnormal, y_train_layer2)

        model_layer2 = RandomForestClassifier(random_state=41, n_estimators=30, max_depth=10)
        model_layer2.fit(X_train_abnormal, y_train_layer2)   # Attention: 如果要用 SMOTE 扩充之后的数据, 要改为 X_train_abnormal_res

        # predict
        y_pred_abnormal = model_layer2.predict(X_test_abnormal)
        y_pred_abnormal_original = np.where(y_pred_abnormal == 0, 1, 2)

        final_preds[test_idx[abnormal_mask]] = y_pred_abnormal_original

    final_preds[test_idx[~abnormal_mask]] = 0

    print(f"\nFlod {flod_idx + 1} Classification Report:")
    print(classification_report(y_test, final_preds[test_idx], target_names=class_name))

# confusion matrix
cm = confusion_matrix(y, final_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
           xticklabels=class_name, 
           yticklabels=class_name)
plt.title("Confusion Matrix (Decision Tree + Random Forest)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

