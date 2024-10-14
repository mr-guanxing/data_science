import numpy as np  
A = np.array([[4, -2],  
              [1,  1]])  
values, vectors = np.linalg.eig(A)  
print("特征值:")  
print(values)  
print("特征向量:")  
print(vectors)