# GMM生成モデルをEMアルゴリズムでパラメータ推定
#----------------------------------------------------
import os, sys
import numpy as np
import matplotlib.pyplot as plt
#自作モジュール
sys.path.append(os.pardir) 
from function import multivariate_normal, gmm, likelihood

#----------------------------------------------------
# データ読み込み
# スクリプトのあるディレクトリ
script_dir = os.path.dirname(__file__) 
# dataの相対path
path = os.path.abspath(f"{script_dir}/../data/old_faithful.txt")
# read 
xs = np.loadtxt(path) # ->(272, 2)
# check the shape
print(xs.shape)

#----------------------------------------------------
# GMM params config
"""
phis: (K,) クラスタの重み
mus: (K,) 各クラスタの多次元ガウスmu(D,)格納
covs: (K,) 各クラスタの多次元ガウスsigma(D,D)格納

K = 2: クラスタ二つ用意
D = 2: 二次元ガウスでfit
"""
# params初期化
phis = np.array([0.5, 0.5])
mus = np.array([[0.0, 50.0], [0.0, 100.0]])
covs = np.array([np.eye(2), np.eye(2)]) # eye(): 単位行列(成分1の対角)

# クラスタ数，サンプル数
K = len(phis)
N = len(xs) # 272

# EMの最大繰り返し数，閾値
MAX_ITERS = 100
THESHOLD = 1e-4

#----------------------------------------------------
# GMM生成モデルparamsをE/Mアルゴリズムで推定

# 終了条件確認
current_likelihood = likelihood(xs, phis, mus, covs)

for iter in range(MAX_ITERS): #iter: (100)E/Mの回数
    #-----------
    # E-step (責任度計算)
    qs = np.zeros((N,K)) # 任意分布qn(z):(272,2)

    for n in range(N): # iter: (272)サンプル毎
        x = xs[n] # サンプルnにおける特徴(2,)

        # 分子(N,K): 特徴毎にスカラを計算
        for k in range(K): # iter: (2)クラスタ毎
            phi, mu, cov = phis[k], mus[k], covs[k]
            qs[n,k] = phi * multivariate_normal(x, mu, cov)
        # 分母(N,): 特徴を周辺化したGMM
        qs[n] /= gmm(x, phis, mus, covs)

    #-----------
    # M-step (疑似完全データ最尤推定)
    qs_sum = qs.sum(axis=0) # (N,K)->(K,)
    for k in range(K): #iter: (2)クラスタ毎
        # 1.phis
        phis[k] = qs_sum[k] / N

        # 2.mus
        c = 0
        for n in range(N): # iter: (272)サンプル毎
            c += qs[n, k] * xs[n]
        mus[k] = c / qs_sum[k]

        # 3.covs 
        c = 0
        for n in range(N): # iter: (272)サンプル毎
            z = xs[n] - mus[k]
            z = z[:, np.newaxis]
            c += qs[n,k] * (z @ z.T)
        covs[k] = c / qs_sum[k]
    
    #-----------
    # 終了判定
    print(f'{current_likelihood:.3f}') # 対数尤度を出力（小数点以下3桁で表示）

    next_likehood = likelihood(xs, phis, mus, covs)
    diff = np.abs(next_likehood - current_likelihood) # 差分の絶対値
    # 満たしていれば終了
    if diff < THESHOLD:
        break
    # 続行
    current_likelihood = next_likehood

#----------------------------------------------------
# 可視化
# visualize
def plot_contour(w, mus, covs):
    x = np.arange(1, 6, 0.1)
    y = np.arange(40, 100, 1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x = np.array([X[i, j], Y[i, j]])

            for k in range(len(mus)):
                mu, cov = mus[k], covs[k]
                Z[i, j] += w[k] * multivariate_normal(x, mu, cov)
    plt.contour(X, Y, Z)

plt.scatter(xs[:,0], xs[:,1])
plot_contour(phis, mus, covs)
plt.xlabel('Eruptions(Min)')
plt.ylabel('Waiting(Min)')
plt.show()

#----------------------------------------------------
# データ生成->phisに基づいて選んだ分布からサンプリング

N = 500
new_xs = np.zeros((N, 2))
for n in range(N):
    k = np.random.choice(2, p=phis)
    mu, cov = mus[k], covs[k]
    new_xs[n] = np.random.multivariate_normal(mu, cov)

#----------------------------------------------------
# visualize
plt.scatter(new_xs[:,0], new_xs[:,1], alpha=0.7, label='generated')
plt.legend()
plt.xlabel('Eruptions(Min)')
plt.ylabel('Waiting(Min)')
plt.show()


