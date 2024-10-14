import numpy as np  
import matplotlib.pyplot as plt  

samples = np.random.randn(100)  
plt.hist(samples, bins=30, edgecolor='black', alpha=0.7, density=True)  
plt.title('Histogram of 100 Samples from Standard Normal Distribution')  
plt.xlabel('Value')  
plt.ylabel('Density')  
  
# 显示标准正态分布的概率密度函数（PDF）作为参考  
from scipy.stats import norm  
xmin, xmax = plt.xlim()  
x = np.linspace(xmin, xmax, 100)  
p = norm.pdf(x, 0, 1)  
plt.plot(x, p, 'k', linewidth=2)  
plt.title('Histogram with Normal Distribution PDF')  

plt.show()
