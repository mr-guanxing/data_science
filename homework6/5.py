import numpy as np  
  

data = np.array([  
    [1,-1,4],  
    [2,1,3],  
    [1,3,-1],  
])  
  
# 计算协方差矩阵  
covariance_matrix = np.cov(data, rowvar=False)  
  
# 打印协方差矩阵  
print(covariance_matrix)