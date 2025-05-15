# 二次元正規分布の可視化
#----------------------------------------------------
import os, sys
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
#自作モジュール
sys.path.append(os.pardir) 
from function import normal,multivariate_normal

#----------------------------------------------------
# 二次元正規分布の描画

# 入力：paramsと2次元特徴
mu = np.array([0.5, -0.2])
cov = np.array([[2.0, 0.3], # 対称行列
                [0.3, 0.5]])

#-----------------
# 定義域と値域：-5から4.9まで0.1ずつ（100個）
xs = ys = np.arange(-5, 5, 0.1) 

# 定義域xsを縦，値域ysを横に要素数だけ広げる
X, Y = np.meshgrid(xs, ys) 
"""
meshgrid([0から4],[0から4])ならば
X = array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4]])

Y = array([[0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1],
          [2, 2, 2, 2, 2],
          [3, 3, 3, 3, 3],
          [4, 4, 4, 4, 4]])

(X[i, j], Y[i, j])で全ての座標をカバーできる
（例：i,j = [3,4]なら X[i,j] = 3, Y[i,j] = 4)
"""
Z = np.zeros_like(X)

# xにX,Y座標の全組み合わせを格納（2特徴）
for i in range(X.shape[0]): 
    for j in range(X.shape[1]): 
        x = np.array([X[i, j], Y[i, j]]) # 全座標値をカバー
        Z[i, j] = multivariate_normal(x, mu, cov) # 代入

# 描画
#新しいFigure（図全体）オブジェクトを作成し、変数 fig に代入
fig = plt.figure()

#figに 1行2列 のサブプロット領域を確保,1番目（左側）に 3D描画用のAxes を作成
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
# 3D形状plot（色付けに Viridis カラーマップを使用）
ax1.plot_surface(X, Y, Z, cmap='viridis')

# 1行2列 のサブプロットを再び指定,今度は 2番目（右側）に 2D描画用 Axes を作成
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
# 等高線plot
ax2.contour(X, Y, Z)

# Figure を画面にレンダリング
plt.show()

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

# 3D描画
ax = plt.axes(projection='3d') # projection='3d'により3d用グラフを指定
ax.plot_surface(X, Y, Z, cmap='viridis') # viridisというカラーマップを使用
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

#----------------------------------------------------
#等高線の描画
x = np.arange(-2, 2, 0.1)
y = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(x, y) # x,yの全組み合わせから座標を得る
Z = X ** 2 + Y ** 2 # 関数値を格納
ax = plt.axes() # 描画領域（Axes オブジェクト）

ax.contour(X, Y, Z) # 等高線の描画
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
