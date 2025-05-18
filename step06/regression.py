# torchで線形回帰（regression）
#----------------------------------------------------
import torch

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

