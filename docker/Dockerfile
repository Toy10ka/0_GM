#構成：GPU ランタイム付きベースイメージを使いつつ、自分の依存だけを追加し、最後に対話シェルを起動

# PyTorch の公式 CUDA 11.7 ランタイム付きイメージを土台にする
# Base image: PyTorch with CUDA runtime (compatible with host driver CUDA 12.x)
# イメージレイヤ内部にはPython（と pip）がすでに入っている
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

# Python の出力をバッファリングせず即時に画面へ流す (no buffering)
ENV PYTHONUNBUFFERED=1

# 以降の COPY／RUN コマンドなどの実行パスを /app に固定
# コンテナ起動後すぐに cd /app 相当の状態になる
WORKDIR /app

# 依存リストのみコンテナ内の /app にコピー (ビルドのたびに変わらないから)
# 後続の依存インストールレイヤをキャッシュ利用できる
COPY requirements.txt ./ 

# ダウンロードしたパッケージのキャッシュを残さずにインストール
#（イメージが小さくなる代わりに毎回ダウンロードが発生）
RUN pip install --no-cache-dir -r requirements.txt

#COPY . ./ <-開発中で頻繁に増減するファイル はマウントしてコンテナに直接同期するべき

#git入れる
#インストール可能なパッケージの一覧更新ー＞依存関係-> index ファイルはインストール後は不要なので削除
RUN apt-get update \ 
  && apt-get install -y git \
  && rm -rf /var/lib/apt/lists/*

# 起動時に自動的にコンテナ内で bash が起動
CMD ["bash"]
