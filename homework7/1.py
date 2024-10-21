import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# 读取CSV文件
csv_file = "./homework7/bike.csv"
data = pd.read_csv(csv_file)

data = data.drop('id', axis=1)

shanghai_data = data[data['city'] == 1]

# 剔除'city'列
shanghai_data = shanghai_data.drop('city', axis=1)

def convert_hour(hour):
    if 6 <= hour <= 18:
        return 1
    else:
        return 0

# 应用函数转换hour列的值
shanghai_data['hour'] = shanghai_data['hour'].apply(convert_hour)

y = shanghai_data['y'].values

# 剔除原先的'y'列
shanghai_data = shanghai_data.drop('y', axis=1)

X = shanghai_data.values

print(X.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import MinMaxScaler

# 初始化MinMaxScaler
scaler = MinMaxScaler()

# 对训练集特征数据进行归一化
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train_scaled, y_train_scaled)

# 输出模型的系数和截距
print("模型系数:", model.coef_)
print("模型截距:", model.intercept_)

from sklearn.metrics import mean_squared_error

# 使用测试集特征数据进行预测
y_pred = model.predict(X_test_scaled)

from math import sqrt

# 计算均方根误差（RMSE）
errors = y_test_scaled - y_pred
# 对差值进行平方
squared_errors = errors ** 2
# 将平方后的差值求和
sum_squared_errors = np.sum(squared_errors)
# 获取样本数量
n = len(y_test)
# 计算MSE
mse = sum_squared_errors / n

print("均方误差（MSE）:", mse)
rmse = sqrt(mse)
# 输出RMSE值
print("均方根误差（RMSE）:", rmse)

