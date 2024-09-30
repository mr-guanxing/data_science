import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sine Wave
x = np.linspace(0, 10, 100)
y = np.sin(x)
sns.lineplot(x=x, y=y)
plt.title('Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('sine_wave_seaborn.png')  
plt.clf()
# plt.show()

# Random Scatter Plot
x = np.random.rand(50)
y = np.random.rand(50)
sns.scatterplot(x=x, y=y)
plt.title('Random Scatter Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('Scatter Plot_seaborn.png') 
plt.clf() 
# plt.show()

# Histogram of Random Data
data = np.random.randn(1000)
sns.histplot(data, bins=30, kde=True, color='blue', alpha=0.7)
plt.title('Histogram of Random Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig('histplot_seaborn.png')  
plt.clf()
# plt.show()
