# 学習手順

WinCLIP は Zero/Few-Shot 学習に対応しており、少数の正常画像から特徴ギャラリーを生成することで推論時の異常検知性能を向上させます。ここではデータセットの配置と学習(ギャラリー生成)の手順をまとめます。

## 1. データセットの準備

1. **MVTec-AD**
   - 公式サイトからデータセットをダウンロードし、リポジトリ直下の `datasets/mvtec_anomaly_detection` に展開します。
2. **VisA**
   - 公式サイトからデータセットを取得し `datasets/VisA_20220922` に展開します。
   - 下記コマンドを実行して PyTorch 用の構造へ変換します。

```bash
python datasets/prepare_visa_public.py
```

変換後 `datasets/VisA_pytorch/1cls` ディレクトリが生成されます。

## 2. 学習の実行

学習は `eval_WinCLIP.py` を `--k-shot` オプション付きで実行することで行います。`k-shot` には学習に使用する画像枚数(0,1,5,10 のいずれか)を指定します。

```bash
python eval_WinCLIP.py \
  --dataset visa \
  --class-name candle \
  --k-shot 5 \
  --gpu-id 0
```

この実行により指定した枚数の正常画像から特徴ギャラリーが構築され、その後テストデータに対する評価が行われます。複数クラスをまとめて処理したい場合は `run_winclip.py` を利用してください。
