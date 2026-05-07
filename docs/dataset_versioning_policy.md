# データセットバージョン管理ポリシー

## 基本原則

Azure Machine Learning の Data asset version は、学習実験の再現性と lineage を保つための単位です。raw データは追記型で保持し、学習に使う単位で curated release を作り、その release を Data asset version として登録します。

## 判断表

| 変更 | 判断 | 理由 |
|---|---|---|
| 画像が追加された | 同一 Data asset の新 version | タスクと schema は同じで、データ範囲だけが変わるため |
| ラベルの誤りを修正した | 同一 Data asset の新 version | 過去の誤りも再現できるようにするため |
| 除外対象を変更した | 同一 Data asset の新 version | 学習対象行が変わるため |
| train / val / test split を変更した | 同一 Data asset の新 version | 評価結果の比較条件が変わるため |
| 後方互換な label を追加した | 同一 Data asset の新 version | 既存タスクの拡張とみなせるため |
| 非互換な label schema に変更した | 新 Data asset | 古いモデル/評価と意味が変わるため |
| 分類から物体検出に変えた | 新 Data asset | タスクと annotation schema が変わるため |
| 顧客/権限/用途が異なる | 新 Data asset | ガバナンス境界を分けるため |
| `uri_folder` から `mltable` に型を変える | 新 Data asset | Azure ML では既存 asset の type 変更は避けるため |

## version 命名

日付ベースを推奨します。

```text
2026.05.07
2026.07.01
2026.10.01
```

日付は「raw データの取得日」ではなく、**curated release として固定した日**に合わせます。

## `@latest` の扱い

| 用途 | `@latest` |
|---|---|
| 探索的な notebook 実験 | 許容 |
| 本番 training | 非推奨 |
| 評価・モデル比較 | 非推奨 |
| 再現実験 | 非推奨 |

最終的な実験記録には、解決済みの Data asset version を必ず残します。

## curated release の不変性

Data asset version 登録後は、対応する curated release folder を変更しません。修正が必要な場合は、次の release folder と Data asset version を作ります。

```text
curated/animal-image-detection/release=2026-07-01/
curated/animal-image-detection/release=2026-07-15/
```

## 統合タイミング

事前統合を標準にします。

| 統合場所 | 推奨度 | 使う場面 |
|---|---:|---|
| data prep / curated release | 高 | 本番、評価、チーム運用 |
| MLTable / manifest | 高 | 複数パス、除外、split、metadata を管理する場合 |
| dataloader | 低 | 研究段階の一時実験のみ |

dataloader は「与えられた 1 つの dataset input を読む」責務に絞ります。