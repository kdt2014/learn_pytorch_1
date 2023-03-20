import matplotlib.pyplot as plt
import numpy as np
# 设置数据
x = np.arange(0, 3, 0.1)
y1 = np.sin(np.pi * x)
y2 = np.cos(np.pi * x)

plt.figure(figsize=(10, 6), facecolor='w', edgecolor='y')

# 绘制第一个子图
plt.subplot(211)
plt.plot(x, y1)
# 绘制第二个子图
plt.subplot(212)
plt.plot(x, y2)

plt.show()