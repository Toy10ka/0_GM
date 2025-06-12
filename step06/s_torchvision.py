# torchvision (pytorchと同じ階層の拡張ライブラリ) 
#----------------------------------------------------
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# torchvision
import torchvision
import torchvision.transforms as transforms
# etc
import matplotlib.pyplot as plt

#----------------------------------------------------
# MNISTデータセットを読み込む

# 前処理
transform = transforms.ToTensor()

dataset = torchvision.datasets.MNIST( # .datasets.MNISTクラスがある
    root = "../data",  # 保存・読み込み位置：実行dirからの相対パス
    train = True,      # Trueならtrain(60k)Falseだとtest(10k)を取得
    transform = transform,  # 前処理を渡す
    download = True    # データがroot に存在しない場合ダウンロード
)

# 最初の画像選択
x, label = dataset[0]

print('size:', len(dataset)) # size: 60000
print('type:', type(x)) # type: <class 'PIL.Image.Image'>
print('label:', label) # label: 5

# 画像を表示する(PILなら.Tensorなら無理)
if dataset.transform == None:
    plt.imshow(x, cmap='gray')
    plt.show()

#----------------------------------------------------
# ミニバッチを使う
dataloader = torch.utils.data.DataLoader( # forでミニバッチを取り出せる
    dataset,
    batch_size=32,
    shuffle=True) # エポックごとのデータをシャッフルする機能

for x, label in dataloader:
    print("x shape:", x.shape) # ->(N, C, H,W)
    print("label shape:", label.shape)
    break 

#----------------------------------------------------
