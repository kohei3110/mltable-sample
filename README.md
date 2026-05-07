# Azure MLTable image dataset versioning sample

このリポジトリは、季節ごとに追加される画像データを Azure Machine Learning の MLTable / Data asset version で管理するための最小サンプルです。

## 結論

新しい季節画像が追加された場合は、原則として **同じ Data asset の新しい version** を作成します。

| 判断ポイント | 推奨 |
|---|---|
| 春データに夏データを追加した | 同一 Data asset の新 version |
| ラベル修正、除外フラグ修正、split 変更をした | 同一 Data asset の新 version |
| ラベル体系に後方互換なクラスを追加した | 同一 Data asset の新 version |
| 分類から物体検出へタスクが変わった | 新しい Data asset |
| ラベル体系が非互換に変わった | 新しい Data asset |
| 顧客、アクセス権限、利用目的が分かれた | 新しい Data asset |
| Data asset type を `uri_folder` から `mltable` に変えたい | 新しい Data asset |

複数の raw パスを学習ジョブや dataloader に直接渡すのではなく、**data prep で curated release を作り、manifest / MLTable で 1 つの論理データセットに統合**します。

## なぜ dataloader で統合しないのか

dataloader に複数の mount path を渡すと、次の問題が起きやすくなります。

- 学習コードが季節・ストレージ・カメラごとのパス構造に依存する
- 実験ごとに「どのパス集合で学習したか」を追跡しづらい
- train / val / test split の再現性が崩れやすい
- 除外理由やラベル schema の変更履歴がコード側に散らばる

このサンプルでは、dataloader は 1 つの `--data` 入力だけを受け取ります。パス統合、除外、split、release 固定は `manifest.csv` と `MLTable` 側に寄せています。

## リポジトリ構成

```text
data_assets/
  animal-image-detection-2026-05-07.yml
  animal-image-detection-2026-07-01.yml

data_prep/
  create_manifest.py
  create_mltable.py
  create_dataset_card.py
  register_data_asset.py

environment/
  conda.yml

jobs/
  train-job.yml
  train-job-direct-mltable.yml

sample_data/
  annotations/animal-images.csv
  raw/...
  curated/animal-image-detection/
    release=2026-05-07/
      MLTable
      manifest.csv
      dataset_card.md
      split_summary.json
    release=2026-07-01/
      MLTable
      manifest.csv
      dataset_card.md
      split_summary.json

src/
  train.py
  dataset.py
```

## サンプル version

| Version | 内容 | Active rows | Excluded rows |
|---|---|---:|---:|
| `2026.05.07` | 春データのみ | 4 | 1 |
| `2026.07.01` | 春 + 夏データ | 7 | 2 |

`2026.05.07` は夏データ追加後も変えません。夏データを含めた release は `2026.07.01` として登録します。

## ローカル smoke test

この確認は Python 標準ライブラリだけで動きます。

```bash
python src/train.py \
  --data sample_data/curated/animal-image-detection/release=2026-07-01 \
  --epochs 1 \
  --output_dir outputs/local-smoke \
  --validate-local-images
```

期待されるポイント:

- `Active samples: 7`
- `Excluded samples: 2`
- `Splits: {'test': 2, 'train': 3, 'val': 2}`
- `outputs/local-smoke/model.txt` と `outputs/local-smoke/dataset_summary.json` が作成される

## curated release の再生成

春のみ release を作る例:

```bash
python data_prep/create_manifest.py \
  --annotations sample_data/annotations/animal-images.csv \
  --raw-root sample_data/raw \
  --output sample_data/curated/animal-image-detection/release=2026-05-07/manifest.csv \
  --include-season spring \
  --strict-local-files

python data_prep/create_mltable.py \
  --release-dir sample_data/curated/animal-image-detection/release=2026-05-07

python data_prep/create_dataset_card.py \
  --manifest sample_data/curated/animal-image-detection/release=2026-05-07/manifest.csv \
  --dataset-card sample_data/curated/animal-image-detection/release=2026-05-07/dataset_card.md \
  --summary sample_data/curated/animal-image-detection/release=2026-05-07/split_summary.json \
  --name animal-image-detection \
  --version 2026.05.07 \
  --release-date 2026-05-07
```

春 + 夏 release を作る場合は `--include-season` を指定せずに全 annotation を対象にするか、必要な季節を複数回指定します。

## Azure ML Data asset 登録

前提:

- Azure Machine Learning workspace がある
- Azure ML CLI v2 が使える
- `image_datalake` datastore がある
- raw 画像が `azureml://datastores/image_datalake/paths/raw/...` に配置されている
- job の Managed Identity または compute managed identity に Storage Blob Data Reader が付与されている

Data asset 登録:

```bash
az ml data create \
  --file data_assets/animal-image-detection-2026-05-07.yml \
  --resource-group <resource-group-name> \
  --workspace-name <workspace-name>

az ml data create \
  --file data_assets/animal-image-detection-2026-07-01.yml \
  --resource-group <resource-group-name> \
  --workspace-name <workspace-name>
```

クラウド運用では、Data asset YAML の `path` をローカルパスではなく curated release の AzureML datastore URI に置き換えるのが一般的です。

例:

```yaml
path: azureml://datastores/image_datalake/paths/curated/animal-image-detection/release=2026-07-01/
```

## Azure ML job 実行

`jobs/train-job.yml` は MLTable 固有の `eval_mount` を使う例です。

```bash
az ml job create \
  --file jobs/train-job.yml \
  --resource-group <resource-group-name> \
  --workspace-name <workspace-name>
```

`jobs/train-job-direct-mltable.yml` は、スクリプト側で `mltable.load()` して materialize したい場合の比較用です。

## MLTable の注意点

このサンプルの `MLTable` は `manifest.csv` を読み、`exclude_reason == ''` の行だけを残します。

画像ファイルを Azure ML data runtime に直接 mount/download させたい場合は、次のどちらかに寄せます。

1. `mltable.from_paths()` で画像パスそのものを MLTable の paths として定義し、パスから `season` や `label` を抽出する
2. manifest の `image_path` 列を `mltable.DataType.to_stream()` で stream 列に変換し、学習コードでは stream オブジェクトを開く

複雑な annotation、split、除外理由を保持する場合は manifest 方式が扱いやすく、フォルダ構造だけで label が決まる単純な画像分類では `from_paths()` 方式も有効です。

## セキュリティと運用のベストプラクティス

- 資格情報、SAS、account key をコードや YAML に書かない
- Azure-hosted job では Managed Identity を使う
- Storage には最小権限を付与する
  - raw 読み取り: Storage Blob Data Reader
  - curated/output 書き込み: Storage Blob Data Contributor
- Storage と compute は同一リージョンに置く
- 本番・評価・再現実験では `@latest` ではなく明示 version を指定する
- Data asset version は不変として扱い、間違えた場合は修正版の新 version を作る
- 古い/誤った asset は削除ではなく archive で隠す

## 参考リンク

- [Create and manage data assets](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-data-assets?view=azureml-api-2)
- [Access data in a job](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-read-write-data-v2?view=azureml-api-2)
- [Working with tables in Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-mltable?view=azureml-api-2)
- [CLI v2 command job YAML schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command?view=azureml-api-2)
- [CLI v2 MLTable YAML schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-mltable?view=azureml-api-2)
- [Data administration](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-administrate-data-authentication?view=azureml-api-2)