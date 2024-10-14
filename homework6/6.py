import numpy as np  
import matplotlib.pyplot as plt  
  
def f(x):  
    return 0.25 * (x - 0.5)**2 + 1  
  
def get_gradient(x):  
    return 0.5 * (x - 0.5)  
  
def update(x, lr):  
    return x - lr * get_gradient(x)  
  

x = 50  
temp = x  
lr = 0.01  
iteration_count = 0 
iteration_history = [] 
function_value_history = []  
  

while True:  
    iteration_count += 1  
    x = update(x, lr)  
    iteration_history.append(iteration_count)   
    function_value_history.append(f(x))  
      
    if abs(x - temp) < 0.01:  
        break  
      
    temp = x  
  
# 绘制迭代过程  
x_values = np.linspace(-1, 3, 400)  # 用于绘制函数曲线的x值范围（此处在可视化中未直接使用）  
y_values = f(x_values)  # 计算对应的函数值（此处在可视化中未直接使用）  
  
plt.scatter(iteration_history, function_value_history, color='red', label='f(x)')  # 绘制梯度下降路径  
plt.xlabel('times')  # 修改横坐标标签为“times”  
plt.ylabel('f(x)')  
plt.legend()  
plt.title('Gradient Descent Iteration Process')  
plt.grid(True)  
plt.show()


