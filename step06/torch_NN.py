# torchで線形回帰（regression）
#----------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#----------------------------------------------------
# ノイズ付きsin関数
torch.manual_seed(0)
x = torch.rand(100, 1)
y = torch.sin(2 * torch.pi * x) + torch.rand(100, 1)

# 2層NN
class TwoNN(nn.Module): # 継承
    """
    構成
    linear - sigmoid - linear
    """
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super().__init__() # 親コンストラクタ

        # 使うレイヤを属性に投げる
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    # forward
    def forward(self, x): # __call__ed
        # 順番にレイヤのforward呼ぶ
        y = self.linear1(x)
        y = F.sigmoid(y)
        y = self.linear2(y)
        return y

#----------------------------------------------------
# 非線形関数をNNでfitting

# hy-para
lr = 0.2
iters = 10000

# model instance
model = TwoNN()

# opt
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# fitting
for i in range(iters):
    
    # forward
    y_pred = model(x)

    # loss & backward
    loss = F.mse_loss(y, y_pred)
    loss.backward()

    # update
    optimizer.step()
    optimizer.zero_grad()

    # display
    if i % 1000 == 0:
        print(loss.item())


