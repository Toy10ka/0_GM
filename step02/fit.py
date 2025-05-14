# 25000個のサンプルから身長の生成モデル（パラメータ推定）を作る
#----------------------------------------------------
import os, sys
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
#自作モジュール
sys.path.append(os.pardir) 
from function import normal

#----------------------------------------------------
# データ読み込み

# スクリプトのあるディレクトリ
script_dir = os.path.dirname(__file__) 
# dataの相対path(絶対の方がいい？)
path = os.path.abspath(f"{script_dir}/../data/height.txt")

#----------------------------------------------------
# 予測分布のパラメータを最尤推定で求める
xs = np.loadtxt(path) #np.loadtxt: 数値前提．ndarrayで格納

# 尤度が最も高くなるパラメータ
mu = np.mean(xs)
sigma = np.std(xs)

#----------------------------------------------------
# 生成（予測分布からのサンプリング）
sample = np.random.normal(mu, sigma)
print(sample)

#----------------------------------------------------
# CDFを出力 ([-∞~x]の積分: t以下となる確率だね)
p1 = norm.cdf(160, mu, sigma) # 160以下
p2 = 1 - norm.cdf(180, mu, sigma) # 180以上

#----------------------------------------------------
# 真の分布と比較
x = np.linspace(150, 190, 1000) # arrangeは幅，linspaceは数を指定
y = normal(x, mu, sigma)

# グラフの描画
plt.hist(xs, bins='auto', density=True)
plt.plot(x, y)
plt.xlabel('Height(cm)')
plt.ylabel('Probability Density')
plt.show()


