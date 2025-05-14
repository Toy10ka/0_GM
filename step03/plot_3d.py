# 3次元グラフを可視化できるらしい
#----------------------------------------------------
import os, sys
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
#自作モジュール
sys.path.append(os.pardir) 
from function import normal,multivariate_normal

#----------------------------------------------------
# X,Yは格子点（引数空間）
X = np.array([[-2, -1, 0, 1, 2],
                [-2, -1, 0, 1, 2],
                [-2, -1, 0, 1, 2],
                [-2, -1, 0, 1, 2],
                [-2, -1, 0, 1, 2]])
Y = np.array([[-2, -2, -2, -2, -2],
                [-1, -1, -1, -1, -1],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2]])
# Zが関数
Z = X ** 2 + Y ** 2

ax = plt.axes(projection='3d') # projection='3d'により3d用グラフを指定
ax.plot_surface(X, Y, Z, cmap='viridis') # viridisというカラーマップを使用
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
