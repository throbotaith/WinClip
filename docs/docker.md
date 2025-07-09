# Docker を使った実行手順

このリポジトリを Docker コンテナ上で動作させるための最小構成の Dockerfile を用意しています。
以下ではイメージのビルドからコンテナ起動までの手順を説明します。

## 1. イメージのビルド

```bash
docker build -t winclip .
```

## 2. コンテナの起動例

データセットをホスト側からマウントすることを想定し、現在のディレクトリをコンテナ内の `/workspace/WinClip` に割り当てます。GPU を使用する場合は `--gpus all` を付与してください。

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/WinClip \
  winclip
```

起動後はコンテナ内で Python スクリプトを実行できます。

