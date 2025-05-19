# torchで線形回帰（regression）
#----------------------------------------------------
import torch
import torch.nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#----------------------------------------------------
# Toy Datasetを作成

# seedを固定
torch.manual_seed(0) 
# 一様分布からサンプリング
x = torch.rand(100, 1) # np.random.rand(100,1)より楽だな
y = 2 * x + 5 + torch.rand(100, 1)

#----------------------------------------------------
# 線形回帰の誤差を正規分布でfitting
# 対数尤度から導く損失関数は平均二乗誤差

# params (マーキング)
W = torch.zeros((1, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 正規分布fitting
def predict(x):
    y = x @ W + b
    return y

# 損失関数(MSE)->nn.funtionにあるからいらない
def mean_squared_error(x0, x1):
    """
    x0: サンプル(N,)
    x1: 予測分布(N,)

    out: loss
    """
    diff = x0 - x1
    N = len(diff)
    return torch.sum(diff ** 2) / N

# ハイパラ
lr = 0.1
iters = 100

#----------------------------------------------------
# 最適化
for i in range(iters):
    # 順伝播->勾配計算
    y_hat = predict(x)
    #loss = mean_squared_error(y, y_hat)
    loss = F.mse_loss(y, y_hat)
    loss.backward()

    # 更新
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    # リセット
    W.grad.zero_()
    b.grad.zero_()

    # たまに出力
    if i % 10 == 0:
        print(loss.item())

# 最適化後params表示
print(loss.item())
print('====')
print('W =', W.item())
print('b =', b.item())

#----------------------------------------------------
# plot
# グラフから切り離してnumpyに変換->散布図描画
plt.scatter(x.detach().numpy(), y.detach().numpy(), s=10)
# 直線描画(detach)
x = torch.tensor([[0.0], [1.0]])
y = W.detach().numpy() * x.detach().numpy() + b.detach().numpy()
plt.plot(x, y, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.show()