# 実データで多次元ガウス予測分布の最尤推定によるparams最適化
#----------------------------------------------------
import os, sys
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
#自作モジュール
sys.path.append(os.pardir) 
from function import normal,multivariate_normal

#----------------------------------------------------
# read data
# スクリプトのあるディレクトリ
script_dir = os.path.dirname(__file__) 
# dataの相対path
path = os.path.abspath(f"{script_dir}/../data/height_weight.txt")
# read 
xs = np.loadtxt(path)
# check the shape
print(xs.shape)

#--------------
# MLE paramas froms data
mu = np.mean(xs, axis=0)
cov = np.cov(xs, rowvar=False)

#--------------
# plot the mailti_normal obtained via MLE.(only first 500)
small_xs = xs[:500]
X, Y = np.meshgrid(np.arange(150, 195, 0.5),
                   np.arange(45, 75, 0.5))
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = np.array([X[i, j], Y[i, j]]) # all conbinations
        Z[i, j] = multivariate_normal(x, mu, cov)

fig = plt.figure()
# 3D
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.plot_surface(X, Y, Z, cmap='viridis')

# contour line
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(small_xs[:,0], small_xs[:,1])
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_xlim(156, 189)
ax2.set_ylim(36, 79)
ax2.contour(X, Y, Z)
plt.show()
