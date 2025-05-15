#ダミーデータで多次元ガウス予測分布の最尤推定params
#----------------------------------------------------
import numpy as np

#----------------------------------------------------
np.random.seed(0) # 乱数生成が常に同じ結果になる(再現性)

N = 10000 # data amount
D = 2 # feature amount

# 入力は一様分布で生成
xs = np.random.rand(N, D)

# mu^（サンプルの平均になる）
mu = np.sum(xs, axis=0) # ->(D,)
mu /= N # (D,) mean of dataの2点

cov = 0

# Sigma^（サンプルの0平均共分散になる）
for n in range(N):
    x = xs[n] # extract the n data　(D,)
    z = x - mu # 平均を引く
    z = z[:, np.newaxis] # 新しい軸を追加(D,1)
    cov += z @ z.T

    cov /= N

# simple method
mu = np.mean(xs, axis=0)
cov = np.cov(xs, rowvar=False) 

print(mu)
print(cov)
# np.cov()：共分散行列を求める
# rowvar：Trueならaxis=0に特徴，Falseならaxis=1に特徴 (今回はaxis=1なのでFalse)
