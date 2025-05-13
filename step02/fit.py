# 25000個のサンプルから身長の生成モデル（パラメータ推定）を作る
#----------------------------------------------------
import os, sys
import numpy as np
import matplotlib.pyplot as plt
#自作モジュール
sys.path.append(os.pardir) 
from step01.function import normal

#----------------------------------------------------
# データ読み込み

# スクリプトのあるディレクトリ
script_dir = os.path.dirname(__file__) 
# dataの相対path(絶対の方がいい？)
path = os.path.abspath(f"{script_dir}/../data/height.txt")

#----------------------------------------------------
# 真の分布のパラメータを推定
xs = np.loadtxt(path) #np.loadtxt: 数値前提．ndarrayで格納
mu = np.mean(xs)
sigma = np.std(xs)

#----------------------------------------------------
# 母集団分布と比較
x = np.linspace(150, 190, 1000) # arrangeは幅，linspaceは数を指定
y = normal(x, mu, sigma)

# グラフの描画
plt.hist(xs, bins='auto', density=True)
plt.plot(x, y)
plt.xlabel('Height(cm)')
plt.ylabel('Probability Density')
plt.show()
