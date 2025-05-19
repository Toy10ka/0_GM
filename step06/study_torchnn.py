# torchで線形回帰（regression）
#----------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#----------------------------------------------------
# nn.Parameter (tensorを継承)
W = nn.Parameter(torch.zeros(1, 1))
b = nn.Parameter(torch.zeros(1))

print(W) # ->tensor([[0.]], requires_grad=True)
print(b) # ->tensor([0.], requires_grad=True)

#----------------------------------------------------
# Mudule継承をやってみよう

class Model(nn.Module): # nn.Module継承
    def __init__(self):
        # 親クラスのコンストラクタ呼び出し
        super().__init__() # super()は親クラスのプロキシインスタンス

        # 追加 (属性にはnn.Parametersを設定)
        self.W = nn.Parameter(torch.zeros(1, 1)) # nn.Para: tensor継承
        self.b = nn.Parameter(torch.zeros(1, 1)) # self.parameters()で呼び出せる

     # forward
    def forward(self, x): # nn.Moduleの空のforwardをオーバーライド
        # y = x @ self.W + self.b
        y = self.linear(x) # self.linear.Wでも見れる
        return y
    
# インスタンス作成
model = Model()

# モデルにあるすべてのパラメータにアクセス
for param in model.parameters():
    print(param)

#----------------------------------------------------
# torch.optimパッケージを使ってみよう

# トイ・データセット
x = torch.rand(100, 1)
y = 5 + 2 * x + torch.rand(100, 1)

# ハイパラ
lr = 0.1
iters = 100

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# 最適化
for i in range(iters): # iters: (100)学習step
    y_hat = model(x) # forward (__call__)
    loss = nn.functional.mse_loss(y, y_hat)
    loss.backward()

    # 更新，リセット
    optimizer.step()
    optimizer.zero_grad()
