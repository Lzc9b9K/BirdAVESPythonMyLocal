# DockerHubのベースとなるイメージを指定
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

# 作業ディレクトリを変更
WORKDIR /app

# aptパッケージの更新
RUN apt-get update -y && apt-get install -y \
        curl \
        git \
        python3 \
        python3-pip \
        wget\
    # && python3 -V \
    && apt-get autoremove -y \
    && apt-get clean \ 
    && rm -rf /var/lib/apt/lists/*

# ローカルPCのファイルをコンテナのカレントディレクトリにコピー
COPY ./requirements.txt ./

# pipのアップデート
RUN pip install --no-cache-dir --upgrade  pip

# pytorch(gpu用)をインストール
RUN  pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# pythonパッケージをインストール
RUN pip install -r requirements.txt
