import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 读取CSV文件
csv_file = r"./homework8/fraudulent.csv"
cleaned_data = pd.read_csv(csv_file)

y = cleaned_data.iloc[:, -1]  # 提取最后一列作为目标变量y
cleaned_data = cleaned_data.drop(cleaned_data.columns[-1], axis=1)


missing_values_percentage = cleaned_data.isnull().mean() * 100
# 设置缺失值比例的阈值
threshold = 30
# 筛选出缺失值比例超过阈值的列
columns_to_remove = missing_values_percentage[missing_values_percentage > threshold].index
# 剔除这些列
cleaned_data = cleaned_data.drop(columns=columns_to_remove)
# 打印剔除列后的DataFrame信息
print("剔除缺失值比例过高的列后的数据信息：")
print(cleaned_data.info())



for column in cleaned_data.columns:
    # 如果该列存在缺失值，添加缺失指示列
    cleaned_data[f'{column}_missing'] = cleaned_data[column].isnull().astype(int)

# 打印添加缺失指示列后的数据信息
print("添加每个特征缺失值指示列后的数据：")
print(cleaned_data.head())

# missingness = cleaned_data.isnull()
# # 对于每个样本，计算每列的缺失值数量
# missing_count_per_sample = missingness.sum(axis=1)
# # 将这些计数转换为列，添加到原始DataFrame中
# cleaned_data['Missingness_Count'] = missing_count_per_sample
# # 打印添加缺失值计数后的数据信息
# print("添加每个样本的缺失值计数的数据信息：")
# print(cleaned_data.info())

from sklearn.impute import SimpleImputer
# 创建一个SimpleImputer对象，使用众数填充
imputer = SimpleImputer(strategy='most_frequent')
# 使用imputer对数据进行拟合和转换
data_imputed = pd.DataFrame(imputer.fit_transform(cleaned_data), columns=cleaned_data.columns)
# 打印填充后的数据信息
print("使用众数填充缺失值后的数据信息：")
print(data_imputed.info())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score

X = data_imputed.iloc[:, :]  # 特征


# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train2,y_train_2 ,X_train2_test, y_train_2_test  = train_test_split(X_train,y_train,test_size=0.2, random_state=42)
# X_train2 训练集 X_train2_test 测试集  X_test验证集


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
f1dic={}
for neibour in range(5,20,2):
    knn_model = KNeighborsClassifier(n_neighbors=neibour)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    print("*"*10)
    print("neighbour:{}".format(neibour))
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix: \n{conf_matrix}")
    print(f"Classification Report: \n{class_report}")
    print(f"F1 Score: {f1}")
    f1dic[neibour] = f1
print(f1dic)
# 建立二分类模型，这里使用逻辑回归
model = LogisticRegression(random_state=42)

# 使用训练集训练模型
model.fit(X_train, y_train)

# 使用测试集测试模型
y_pred = model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

f1 = f1_score(y_test, y_pred, average='binary')

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix: \n{conf_matrix}")
print(f"Classification Report: \n{class_report}")
print(f"F1 Score: {f1}")
