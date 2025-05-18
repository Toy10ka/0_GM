#関数を定義したい気持ち
#----------------------------------------------------
import numpy as np

#----------------------------------------------------
#ガウス分布
def normal(x, mu=0, sigma=1):
    """
    x: 確率変数
    mu, sigma: param
    mu=0, sigma=1なら標準正規分布
    """
    y = 1 / (np.sqrt(2*np.pi) * sigma) * np.exp(-(x-mu)**2 / (2*sigma**2))
    return y

#----------------------------------------------------
#多次元ガウス分布
def multivariate_normal(x, mu, cov):
    """
    x: 特徴(D,) 
    mu: (D,)
    sigma: (D, D)

    y: スカラ
    """
    # 構成要素
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)

    # 係数
    z = 1 / np.sqrt((2*np.pi) ** D * det)

    # exp
    y = z * np.exp((x-mu).T @ inv @ (x-mu) / -2.0) # @: np.dot

    return y

#----------------------------------------------------
# 混合ガウスモデル
def gmm(x, phis, mus, covs):
    """
    x: 特徴(D,) 
    phis: カテゴリカル配列(K,)
    mus: (len(μ),)
    sigmas: (len(sigma), )

    y: スカラ
    """
    K = len(phis)
    y = 0
    
    for k in range(K):
        # 分布を選び，idxとparamsを格納
        phi, mu, cov = phis[k], mus[k], covs[k]
        # 重み付き加算
        y += phi * multivariate_normal(x, mu, cov)
    return y 

#----------------------------------------------------
# EMalgの閾値(対数尤度平均)
def likelihood(xs, phis, mus, covs):
    """
    xs: サンプル(D,) 
    mus: (K,)
    covs: (K,)

    y: スカラ
    """
    eps = 1e-8 # log(0)を防ぐための微小値
    L = 0
    N = len(xs)
    # 閾値＝サンプル毎の対数尤度の平均
    for x in xs:
        y = gmm(x, phis, mus, covs)
        L += np.log(y + eps)
    return L / N

