import numpy as np
import os, sys
import matplotlib.pyplot as plt
#自作モジュール
sys.path.append(os.pardir) 
from function import normal,multivariate_normal
# ======= 学習済みのパラメータ =======
mus = np.array([[2.0, 54.50],
                [4.3, 80.0]])
covs = np.array([[[0.07, 0.44],
                [0.44, 33.7]],
                [[0.17, 0.94],
                [0.94, 36.00 ]]])
phis = np.array([0.35, 0.65]) # 総和は1である必要アリ
# =================================
def sample():
    # phis で指定された確率分布に従って、0、1 のいずれかをサンプリング
    z = np.random.choice(2, p=phis) 
    # どっちのparams 
    mu, cov = mus[z], covs[z]
    x = np.random.multivariate_normal(mu, cov) # npの組み込み関数（D,）
    return x

# data一個一個をnp.multi_normalに入れていく
N = 500
xs = np.zeros((N, 2))
for i in range(N):
    xs[i] = sample() # -> (N,)

plt.scatter(xs[:,0], xs[:,1], color='orange', alpha=0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
