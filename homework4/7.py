import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('linspace_matplot.png')  
plt.clf()
# plt.show()


x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y)
plt.title('Random Scatter Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('Random Scatter_matplot.png')  
plt.clf()
# plt.show()


data = np.random.randn(1000)

plt.hist(data, bins=30, alpha=0.7)
plt.title('Histogram of Random Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig('sine_wave_matplot.png')  
plt.clf()
# plt.show()