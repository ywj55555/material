import math
import numpy as np
import matplotlib.pyplot as plt

U = [450, 515, 560, 580, 640, 659, 680, 700, 725, 777, 800, 852]
# u = 500   # 均值μ
for i,u in enumerate(U):
    sig = math.sqrt(400)  # 标准差δ
    # nums = 100
    if i==0:
        # nums
        x = np.linspace(u - 4*sig, u + 8*sig, 50)   # 定义域
    elif i == len(U)-1:
        x = np.linspace(u - 8 * sig, u + 4 * sig, 50)
    else:
        x = np.linspace(u - 8 * sig, u + 8 * sig, 50)  # 定义域
    y = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig) # 定义曲线函数
    y = y*3500
    plt.plot(x, y, "g", linewidth=2)    # 加载曲线

# 全色通道
sig = math.sqrt(100000)  # 标准差δ
# nums = 100
u = 650
x = np.linspace(u - 0.8 * sig, u + 0.8 * sig, 50)  # 定义域
y = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig) # 定义曲线函数
y = y*50000
plt.plot(x, y, "r", linewidth=2)    # 加载曲线

plt.grid(True)  # 网格线
plt.savefig('sensor_normal.png')
plt.show()  # 显示