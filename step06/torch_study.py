# pytorchを使わせていただこう
#----------------------------------------------------
import torch

#----------------------------------------------------
# pytorchの勉強する
print(torch.__version__)

x = torch.tensor(5.0)
y = 3 * x ** 2
print(y) # 75.

# 自動微分で遊ぶ
# 勾配が欲しい変数xをマーキング（計算グラフの葉）
x = torch.tensor(5.0, requires_grad=True) 

# 演算子ごとにFunctionレイヤが実装されているので，requires_grad=Trueならば
# そのインスタンスがgrad_fn に格納される（それを逆にbackwardで辿る）
y = 3 * x ** 2 # 計算グラフが構築

# 呼ばれるとグラフを逆にたどって勾配計算とグラフ破棄
y.backward() 
print(x.grad) # 30.

#----------------------------------------------------
# 自動微分を利用して勾配を出そう

# 対象の関数
def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y

# マーキング
x0 = torch.tensor(0.0, requires_grad=True)
x1 = torch.tensor(2.0, requires_grad=True)

# 定式化するだけで各演算子のレイヤがgrad_fnに記録されていく
y = rosenbrock(x0, x1)

# backwardと言うだけで計算グラフを逆にたどって，勾配.gradが流れ着いてくる
y.backward()

# 生まれた勾配
print(x0.grad, x1.grad) # tensor(-2.) tensor(400.)

#----------------------------------------------------
# 勾配降下法による最小値の導出

# マーキング
x0 = torch.tensor(0.0, requires_grad=True)
x1 = torch.tensor(2.0, requires_grad=True)

# ハイパラ
lr = 0.001 
iters = 10000 

for i in range(iters): # iter: (10000)学習回数
    # たまに値を見せてほしい
    if i % 1000 == 0:
        print(x0.item(), x1.item()) # item: 対象がスカラならtensor->python数値へ

    # 勾配自動計算
    y = rosenbrock(x0, x1)
    y.backward()

    # 降下法
    x0.data -= lr * x0.grad.data # .data: 生の数値データにアクセス(非推奨)
    x1.data -= lr * x1.grad.data # with torch.no_grad():とかが推奨

    # 勾配リセット (x0.grad や x1.grad には、微分が加算されていく)
    x0.grad.zero_()
    x1.grad.zero_()

# 最小値表示
print(x0.item(), x1.item())




