# 独自データセットでの学習

20 枚の正常画像と 300 枚の異常画像のように，小規模なデータを用いて特徴ギャラリーを生成する場合は `datasets/custom` ディレクトリに以下の構成で画像を配置します。

```
datasets/custom/
  train/
    good/      # 正常画像 (20 枚)
  test/
    anomaly/   # 異常画像 (300 枚)
```

`eval_WinCLIP.py` で `--dataset custom` を指定するとこのフォルダ構造を読み込みます。`--k-shot` には使用する正常画像枚数を設定できます。

```bash
python eval_WinCLIP.py \
  --dataset custom \
  --class-name custom \
  --k-shot 20 \
  --gpu-id 0
```

これにより 20 枚の正常画像からギャラリーを生成し，300 枚の異常画像で評価を行います。
