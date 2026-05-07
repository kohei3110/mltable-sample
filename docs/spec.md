# 仕様書：Azure Machine Learning における画像データセット管理サンプル

## 1. 目的

本サンプルは、Azure Machine Learning を用いた画像 AI 開発において、以下の課題を解決するためのリファレンス実装を提供する。

* 季節ごと・期間ごとに追加される画像データを、再現性のある形で管理する
* 複数フォルダ・複数ストレージに分散した画像を、学習コード側で複雑に扱わず、1 つの論理データセットとして利用する
* Azure ML Data asset のバージョン管理を用いて、どのデータでどのモデルを学習したかを追跡できるようにする
* データ統合処理を dataloader に持たせすぎず、データ準備パイプライン / MLTable / manifest 側に寄せる

Azure ML の Data asset は、長いストレージ URI を覚えずに、`azureml:<data_asset_name>:<version>` のような名前付きバージョンでデータを参照するための仕組みとして利用できる。([Microsoft Learn][1])

---

## 2. 対象シナリオ

### 2.1 想定ユースケース

画像分類または物体検出モデルの学習を想定する。

例：

* 季節ごとに動物画像が追加される
* 撮影元が複数ある
* 画像とアノテーションファイルが別々の場所に保存されている
* 学習時には、最新版または特定バージョンのデータセットを指定して実験したい
* 過去の学習結果を、同じデータセットバージョンで再現したい

---

## 3. 設計方針

## 3.1 基本方針

本サンプルでは、以下の設計方針を採用する。

| 項目          | 方針                                       |
| ----------- | ---------------------------------------- |
| Raw データ     | 原則として追記型で保存する                            |
| Curated データ | 学習に使う単位で release フォルダを作成する               |
| データセット管理    | Azure ML Data asset の version として登録する    |
| 複数パス統合      | dataloader ではなく MLTable / manifest で管理する |
| 学習ジョブ入力     | 1 つの Data asset version のみを渡す            |
| 本番・評価実験     | `@latest` ではなく明示的な version を指定する         |
| 一時実験        | 必要に応じて `@latest` を許容する                   |

Azure ML では Data asset の種類として `uri_file`、`uri_folder`、`mltable` を利用できる。画像フォルダ単体であれば `uri_folder`、複数パス・注釈ファイル・split 情報を論理的にまとめる場合は `mltable` を利用する。([Microsoft Learn][1])

---

# 4. サンプルアーキテクチャ

## 4.1 全体構成

```text
画像アップロード
  ↓
Azure Blob Storage / ADLS Gen2
  ↓
raw 領域に追記保存
  ↓
データ準備ジョブ
  - QC
  - 重複除外
  - ラベル整合性チェック
  - train / val / test split 作成
  - manifest 作成
  - MLTable 作成
  ↓
curated release 作成
  ↓
Azure ML Data asset version 登録
  ↓
Azure ML training job
  ↓
モデル・メトリクス・データリネージ記録
```

---

## 4.2 ストレージ構成

```text
azureml://datastores/image_datalake/paths/

raw/
  source=camera-a/
    season=spring/
      collected_date=2026-04-01/
        img001.jpg
        img002.jpg

  source=camera-b/
    season=summer/
      collected_date=2026-07-01/
        img101.jpg
        img102.jpg

annotations/
  label_schema=v1/
    batch=2026-05-07/
      annotations.jsonl

curated/
  animal-image-detection/
    release=2026-05-07/
      MLTable
      manifest.csv
      dataset_card.md
      split_summary.json
```

---

# 5. データセットバージョン管理ルール

## 5.1 Data asset 名

```text
animal-image-detection
```

## 5.2 バージョン命名規則

日付ベースを推奨する。

```text
2026.05.07
2026.06.01
2026.07.15
```

または、SemVer 形式も利用可能。

```text
1.0.0
1.1.0
2.0.0
```

## 5.3 バージョン作成判断

| 変更内容                           | 対応                       |
| ------------------------------ | ------------------------ |
| 画像が追加された                       | 同一 Data asset の新 version |
| ラベル修正を行った                      | 同一 Data asset の新 version |
| train / val / test split を変更した | 同一 Data asset の新 version |
| ラベル体系を後方互換で追加した                | 同一 Data asset の新 version |
| ラベル体系が非互換に変わった                 | 新しい Data asset           |
| タスクが分類から物体検出に変わった              | 新しい Data asset           |
| 顧客・アクセス権限・用途が完全に異なる            | 新しい Data asset           |

---

# 6. Manifest 仕様

## 6.1 目的

manifest は、画像ファイルとラベル、split、メタデータを紐づけるための管理ファイルである。
学習コードは複数の raw パスを直接意識せず、manifest を読み取って学習データを構築する。

---

## 6.2 `manifest.csv` スキーマ

```csv
image_path,label,split,season,source,collected_date,label_schema_version,annotation_batch,exclude_reason
azureml://datastores/image_datalake/paths/raw/source=camera-a/season=spring/collected_date=2026-04-01/img001.jpg,cat,train,spring,camera-a,2026-04-01,v1,2026-05-07,
azureml://datastores/image_datalake/paths/raw/source=camera-b/season=summer/collected_date=2026-07-01/img101.jpg,dog,val,summer,camera-b,2026-07-01,v1,2026-05-07,
```

## 6.3 カラム定義

| カラム                    | 必須 | 説明                                        |
| ---------------------- | -: | ----------------------------------------- |
| `image_path`           | 必須 | 画像ファイルの AzureML datastore URI             |
| `label`                | 必須 | 画像分類ラベル、または主ラベル                           |
| `split`                | 必須 | `train` / `val` / `test`                  |
| `season`               | 任意 | `spring` / `summer` / `autumn` / `winter` |
| `source`               | 任意 | 撮影元、施設、カメラ、データ提供元など                       |
| `collected_date`       | 任意 | 画像取得日                                     |
| `label_schema_version` | 必須 | ラベル体系のバージョン                               |
| `annotation_batch`     | 任意 | アノテーション実施単位                               |
| `exclude_reason`       | 任意 | 除外理由。空欄なら利用対象                             |

---

# 7. MLTable 仕様

## 7.1 目的

MLTable は、manifest や複数の画像パスを Azure ML のジョブ入力として評価可能にするために利用する。

Azure ML では、MLTable 用の `eval_mount` / `eval_download` を使うことで、MLTable を評価した結果を compute target に mount / download できる。これにより、画像が複数ストレージや複数コンテナに分散していても、学習ジョブ側では評価済みの入力として扱える。([Microsoft Learn][2])

## 7.2 `MLTable` ファイル例

```yaml
paths:
  - file: ./manifest.csv

transformations:
  - read_delimited:
      delimiter: ','
      encoding: utf8
      header: all_files_same_headers
  - filter: "exclude_reason == ''"
```

注意点として、Azure ML はファイル名として `MLTable` を期待するため、`MLTable.yaml` や `MLTable.yml` にリネームしない。([Microsoft Learn][1])

## 7.3 画像パス列の扱い

本サンプルでは、`manifest.csv` に以下の 2 種類のパス列を持たせる。

| カラム | 用途 |
|---|---|
| `image_path` | Azure ML / Azure Storage 上の本番用 URI |
| `local_image_path` | このリポジトリの smoke test 用ローカル相対パス |

クラウド運用では `image_path` を正とし、`local_image_path` は不要である。ローカルでサンプルを動かすためだけに `local_image_path` を含めている。

画像ファイルそのものを Azure ML data runtime に評価・mount/download させたい場合は、以下のいずれかを使う。

* `mltable.from_paths()` で画像ファイル群を MLTable の paths として定義する
* manifest の `image_path` 列を `mltable.DataType.to_stream()` で stream 列に変換する

複雑な annotation、除外理由、split、dataset card を運用する場合は manifest 方式が扱いやすい。一方で、フォルダ構造だけで label が決まる単純な画像分類では `from_paths()` 方式も有効である。

---

# 8. Azure ML Data asset 登録仕様

## 8.1 Data asset 定義

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json
name: animal-image-detection
version: 2026.05.07
type: mltable
path: azureml://datastores/image_datalake/paths/curated/animal-image-detection/release=2026-05-07/
description: Image dataset for animal image detection. Includes spring and summer images.
tags:
  task: image-detection
  label_schema_version: v1
  release_date: 2026-05-07
  data_stage: curated
```

## 8.2 登録コマンド

```bash
az ml data create \
  --file data/animal-image-detection.yml \
  --resource-group <resource-group-name> \
  --workspace-name <workspace-name>
```

---

# 9. 学習ジョブ仕様

## 9.1 学習ジョブの入力方針

学習ジョブには、複数の raw パスを個別に渡さない。
必ず 1 つの Data asset version を渡す。

悪い例：

```bash
python train.py \
  --spring_path /mnt/spring \
  --summer_path /mnt/summer \
  --winter_path /mnt/winter
```

良い例：

```bash
python train.py \
  --data ${{inputs.training_data}}
```

Azure ML の command job YAML では、登録済み Data asset を `azureml:<data_name>:<data_version>` または `azureml:<data_name>@latest` 形式で参照できる。ただし、再現性が必要な学習では明示的な version 指定を推奨する。([Microsoft Learn][3])

---

## 9.2 `train-job.yml` 例

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
type: command

display_name: train-animal-image-model
experiment_name: animal-image-detection

code: ./src

command: >-
  python train.py
  --data ${{inputs.training_data}}
  --epochs 10
  --output_dir ${{outputs.model_output}}

inputs:
  training_data:
    type: mltable
    path: azureml:animal-image-detection:2026.05.07
    mode: eval_mount

outputs:
  model_output:
    type: uri_folder
    mode: upload

environment: azureml:animal-image-training-env:1
compute: azureml:gpu-cluster
```

Azure ML の入力モードには `ro_mount`、`download`、`direct`、`eval_mount`、`eval_download` などがあり、`eval_mount` / `eval_download` は MLTable 固有のモードである。([Microsoft Learn][3])

---

# 10. 学習コード仕様

## 10.1 `train.py` の責務

`train.py` は以下のみを担当する。

* Azure ML から渡された 1 つの `--data` パスを受け取る
* manifest を読み込む
* `split=train` / `split=val` / `split=test` に分割する
* 画像とラベルをロードする
* 学習を実行する
* モデルを保存する

## 10.2 `train.py` サンプル

```python
import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    manifest_path = os.path.join(args.data, "manifest.csv")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.csv not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required_columns = [
        "image_path",
        "label",
        "split",
        "label_schema_version",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # TODO:
    # - Load images
    # - Build dataset
    # - Train model
    # - Save model

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "model.txt"), "w") as f:
        f.write("dummy model artifact")


if __name__ == "__main__":
    main()
```

---

# 11. データ準備パイプライン仕様

## 11.1 処理内容

データ準備パイプラインでは、以下を実行する。

| ステップ | 処理内容                           |
| ---- | ------------------------------ |
| 1    | raw データの一覧取得                   |
| 2    | アノテーションファイル読み込み                |
| 3    | 画像ファイル存在チェック                   |
| 4    | ラベル体系チェック                      |
| 5    | 重複画像チェック                       |
| 6    | 除外対象の判定                        |
| 7    | train / val / test split 作成    |
| 8    | manifest.csv 生成                |
| 9    | MLTable 生成                     |
| 10   | dataset_card.md 生成             |
| 11   | Azure ML Data asset version 登録 |

---

## 11.2 データ準備ジョブの出力

```text
curated/
  animal-image-detection/
    release=2026-05-07/
      MLTable
      manifest.csv
      dataset_card.md
      split_summary.json
```

---

# 12. Dataset Card 仕様

## 12.1 `dataset_card.md` 例

```markdown
# Dataset Card: animal-image-detection:2026.05.07

## Overview

This dataset is used for animal image detection training.

## Dataset Version

- Data asset name: animal-image-detection
- Version: 2026.05.07
- Label schema version: v1
- Release date: 2026-05-07

## Data Sources

| Source | Season | Count |
|---|---|---:|
| camera-a | spring | 1,200 |
| camera-b | summer | 980 |

## Splits

| Split | Count |
|---|---:|
| train | 1,700 |
| val | 300 |
| test | 180 |

## Label Distribution

| Label | Count |
|---|---:|
| cat | 1,100 |
| dog | 1,080 |

## Known Limitations

- Autumn and winter images are not included in this version.
- Low-light images are underrepresented.
- Label schema is v1 and may be revised in future versions.
```

---

# 13. GitHub リポジトリ構成

```text
azureml-image-dataset-versioning-sample/
  README.md

  data_assets/
    animal-image-detection.yml

  jobs/
    train-job.yml

  src/
    train.py
    dataset.py
    utils.py

  data_prep/
    create_manifest.py
    create_mltable.py
    create_dataset_card.py
    register_data_asset.py

  sample_data/
    manifest.csv
    MLTable
    dataset_card.md

  docs/
    architecture.md
    dataset_versioning_policy.md
    operation_guide.md
```

---

# 14. README に記載する実行手順

## 14.1 前提

* Azure Machine Learning workspace が作成済みであること
* Azure ML CLI v2 が利用可能であること
* 画像データが Azure Storage / datastore に配置されていること
* `image_datalake` という datastore が Azure ML workspace に登録されていること

## 14.2 Data asset 登録

```bash
az ml data create \
  --file data_assets/animal-image-detection.yml \
  --resource-group <resource-group-name> \
  --workspace-name <workspace-name>
```

## 14.3 学習ジョブ実行

```bash
az ml job create \
  --file jobs/train-job.yml \
  --resource-group <resource-group-name> \
  --workspace-name <workspace-name>
```

---

# 15. 運用設計

## 15.1 データ追加時の運用

新しい季節画像が追加された場合は、以下の流れで運用する。

```text
新規画像を raw に追加
  ↓
アノテーション追加
  ↓
データ準備パイプライン実行
  ↓
新しい release フォルダ作成
  ↓
新しい Data asset version 登録
  ↓
学習ジョブで version を明示指定
```

例：

```text
animal-image-detection:2026.05.07
animal-image-detection:2026.07.01
animal-image-detection:2026.10.01
```

---

## 15.2 学習時のバージョン指定

本番・評価用途では、必ず明示的に version を指定する。

```yaml
path: azureml:animal-image-detection:2026.05.07
```

実験用途では、必要に応じて `@latest` を利用してもよい。

```yaml
path: azureml:animal-image-detection@latest
```

ただし、最終的な評価・比較・再現実験では、実際に解決された Data asset version を記録する。

---

# 16. 受け入れ基準

本サンプルは、以下を満たすことを完了条件とする。

| No | 受け入れ基準                                                     |
| -: | ---------------------------------------------------------- |
|  1 | `manifest.csv` から画像パス、ラベル、split を読み込める                     |
|  2 | `MLTable` を含む curated release フォルダを作成できる                   |
|  3 | Azure ML Data asset として `mltable` type で登録できる              |
|  4 | 学習ジョブが `azureml:<data_name>:<version>` で Data asset を参照できる |
|  5 | 学習コードが複数 raw パスを意識せず、1 つの `--data` 入力だけで動作する               |
|  6 | 新しい季節データを追加した際、新しい Data asset version として登録できる             |
|  7 | 過去の Data asset version を指定して再実行できる                         |
|  8 | dataset card にデータ件数、split、ラベル分布、制約事項が記録される                 |

---

# 17. 最小 MVP スコープ

まず作るなら、以下の MVP がよいです。

```text
MVP 1:
- sample manifest.csv
- MLTable
- Data asset YAML
- train-job.yml
- train.py
- README

MVP 2:
- create_manifest.py
- create_dataset_card.py
- version naming policy
- split summary output

MVP 3:
- Azure ML pipeline 化
- data prep component
- training component
- evaluation component
- model registration
```

[1]: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-data-assets?view=azureml-api-2 "Create Data Assets - Azure Machine Learning | Microsoft Learn"
[2]: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-read-write-data-v2?view=azureml-api-2 "Access data in a job - Azure Machine Learning | Microsoft Learn"
[3]: https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command?view=azureml-api-2 "CLI (v2) command job YAML schema - Azure Machine Learning | Microsoft Learn"
