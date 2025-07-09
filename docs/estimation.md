# 推論手順

ここでは学習済みモデルを用いた推論(評価)の実行方法を説明します。

## 1. 依存ライブラリのインストール

コンテナを使用している場合は `install.sh` を実行して Python の依存ライブラリをインストールします。

```bash
bash install.sh
```

## 2. データセットの配置

- MVTec-AD データセット: `datasets/mvtec_anomaly_detection` に展開してください。
- VisA データセット: `datasets/VisA_20220922` に展開後、以下を実行して変換します。

```bash
python datasets/prepare_visa_public.py
```

変換後 `datasets/VisA_pytorch/1cls` 以下にデータが作成されます。

## 3. 推論の実行

特定クラスに対して推論を実行するには `eval_WinCLIP.py` を使用します。
例として VisA の `candle` クラスを処理する場合は以下のように実行します。

```bash
python eval_WinCLIP.py --dataset visa --class-name candle --gpu-id 0
```

複数クラスをまとめて評価したい場合は `run_winclip.py` を実行してください。

```bash
python run_winclip.py
```

結果は `result_winclip` ディレクトリ以下に保存されます。
