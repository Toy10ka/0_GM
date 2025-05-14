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
