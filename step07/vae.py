# vaeを実装
#----------------------------------------------------
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

#----------------------------------------------------
# ハイパラ
input_dim = 784
hidden_dim = 200
latent_dim = 20 # 潜在変数ベクトルzの次元数
epochs = 20
learning_rate = 3e-4
batch_size = 32

#-------------------
# エンコーダ(二層NN)
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim): # 自動追跡
        super().__init__()
        # 1層目
        self.linear = nn.Linear(input_dim, hidden_dim)
        # mu, sigmaの枝分かれ出力
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        """
        x: (N,D) 

        mu: (N,z)
        sigma: (N,z)
        """
        # h1
        h = self.linear(x)
        h = F.relu(h)
        # output
        mu = self.linear_mu(h)
        # sigmaを正にするため
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma
    
#-------------------
# デコーダ(二層NN)
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        # 1
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        # 2
        self.linear2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        """
        z: (N,z) 

        x_hat: (N,x)
        """
        h = self.linear1(z) # error!!
        h = F.relu(h)
        h = self.linear2(h)
        x_hat = F.sigmoid(h)
        return x_hat
    
#----------------------------------------------------
# 変数変換サンプリング関数
def reparameterize(mu, sigma):
    # sigmaと同じ形状のテンソル
    eps = torch.randn_like(sigma)
    z = mu + eps * sigma
    return z

#-------------------
# VAEクラス
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim): 
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def get_loss(self, x):
        """
        x: (N,D) 

        loss: 平均損失値(torch.Tensor)
        """
        mu, sigma = self.encoder(x)
        z = reparameterize(mu, sigma)
        x_hat = self.decoder(z)

        batch_size = len(x)
        # 損失関数（二項それぞれ計算）
        L1 = F.mse_loss(x_hat, x, reduction="sum") # sumにすると総和
        L2 = - torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)
        return (L1 + L2) / batch_size
    
#----------------------------------------------------
# torchvisionでデータセット読み込み

# 変形指示
transform = transforms.Compose([ # 前処理のラッパ
    transforms.ToTensor(),  # 画像→テンソル (B,C,H,W)
    transforms.Lambda(torch.flatten) # Lambda: 任意関数をラップ
])

# データセットのインスタンス化
dataset = datasets.MNIST(
    root = "../data", # ダウンロード先
    train = True, # Falseだとテストセット
    download = True, # ローカルに無ければ自動ダウンロード
    transform = transform
)

# データローダ
dataloader = torch.utils.data.DataLoader( # datasetをpytorchにイテレータとして渡す
    dataset, # 元になるデータセットインスタンス
    batch_size = batch_size, # 1batchに何個入れるか
    shuffle = True # 各エポックでサンプル順をシャッフル
)
#----------------------
# modelとoptimizer
model = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
losses = []

#----------------------
# train
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0
    # 1train
    for x, label in dataloader:
        # リセット
        optimizer.zero_grad()
        # 損失→勾配
        loss = model.get_loss(x)
        loss.backward()
        # 更新
        optimizer.step()

        loss_sum += loss.item() # tensor->pythonの数値型
        cnt += 1

    # epoch毎に損失の平均を計算
    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(loss_avg)

#----------------------
# 画像生成