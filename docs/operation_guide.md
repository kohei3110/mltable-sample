# 運用ガイド: 季節データ追加時の流れ

## 1. raw 領域へ追記

季節ごとの新規画像は raw 領域に追記します。既存ファイルは上書きしません。

```text
raw/
  source=camera-b/
    season=summer/
      label=cat/
        collected_date=2026-07-01/
          cat_summer_001.ppm
```

## 2. annotation を追加

`sample_data/annotations/animal-images.csv` のような annotation 管理ファイルに、画像パス、label、split、season、source、schema version、除外理由を記録します。

## 3. curated release を作成

release folder を作ります。

```text
curated/animal-image-detection/release=2026-07-01/
```

作成物:

- `manifest.csv`
- `MLTable`
- `dataset_card.md`
- `split_summary.json`

## 4. manifest を検証

確認項目:

- 必須列が存在する
- `split` は `train` / `val` / `test` のみ
- `exclude_reason` が空の行だけが学習対象になる
- label schema version が意図したものか
- 新しい季節の件数が dataset card に反映されているか

## 5. Data asset version として登録

例:

```text
animal-image-detection:2026.05.07
animal-image-detection:2026.07.01
```

同じ asset name に対して version を増やします。

## 6. training job で明示 version を指定

```yaml
inputs:
  training_data:
    type: mltable
    path: azureml:animal-image-detection:2026.07.01
    mode: eval_mount
```

評価や本番では `azureml:animal-image-detection@latest` を使いません。

## 7. lineage を確認

Azure ML Studio の Data asset 画面で、その Data asset version を消費した jobs を確認できます。モデル評価で問題が起きた場合は、job → input data asset version → curated release → raw/annotation の順に追跡します。

## 8. 誤登録時の扱い

Data asset version は削除前提で運用しません。

- path が間違っていた: 正しい path の新 version を作る
- 名前/type が間違っていた: archive して、新しい Data asset を作る
- dataset card が間違っていた: 修正版の curated release と新 version を作る

## 9. performance 注意点

画像のような小さいファイルが大量にある場合は、Storage request 数がボトルネックになりやすいです。

対策候補:

- Storage と compute を同一リージョンに置く
- `eval_mount` / `eval_download` を使い、独自 Python downloader を避ける
- 複数 epoch では mount cache を有効化する
- 数百万ファイル規模では tar/zip shard や Premium Blob を検討する
- Storage account の egress/request limit を監視する