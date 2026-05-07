from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
from typing import Any


OUTPUT_COLUMNS = [
    "image_path",
    "local_image_path",
    "label",
    "split",
    "season",
    "source",
    "collected_date",
    "label_schema_version",
    "annotation_batch",
    "exclude_reason",
]

VALID_SPLITS = {"train", "val", "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a versioned image manifest for an Azure MLTable release.")
    parser.add_argument("--annotations", required=True, help="Annotation CSV with relative_path, label, season, and metadata.")
    parser.add_argument("--raw-root", required=True, help="Local raw data root. Used to generate local_image_path values.")
    parser.add_argument("--output", required=True, help="Output manifest.csv path under a curated release folder.")
    parser.add_argument(
        "--azureml-uri-prefix",
        default="azureml://datastores/image_datalake/paths/raw",
        help="AzureML datastore URI prefix for production image_path values.",
    )
    parser.add_argument("--include-season", action="append", default=[], help="Include only this season. Repeatable.")
    parser.add_argument("--split-seed", default="animal-image-detection-v1", help="Seed for deterministic split assignment.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--strict-local-files", action="store_true", help="Fail if an active local image is missing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = Path(args.annotations)
    raw_root = Path(args.raw_root)
    output = Path(args.output)

    rows = read_annotations(annotations)
    rows = filter_rows(rows, include_seasons=set(args.include_season))
    manifest_rows = [build_manifest_row(row, raw_root, output.parent, args.azureml_uri_prefix, args) for row in rows]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Wrote {len(manifest_rows)} manifest rows to {output}")


def read_annotations(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"relative_path", "label", "season", "source", "collected_date", "label_schema_version"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Annotation file is missing columns: {sorted(missing)}")
        return [{key: clean(value) for key, value in row.items()} for row in reader]


def filter_rows(rows: list[dict[str, str]], include_seasons: set[str]) -> list[dict[str, str]]:
    if not include_seasons:
        return rows
    return [row for row in rows if row.get("season") in include_seasons]


def build_manifest_row(
    row: dict[str, str],
    raw_root: Path,
    release_dir: Path,
    azureml_uri_prefix: str,
    args: argparse.Namespace,
) -> dict[str, str]:
    relative_path = normalize_relative_path(row["relative_path"])
    local_file = raw_root / relative_path
    exclude_reason = clean(row.get("exclude_reason"))

    if args.strict_local_files and not exclude_reason and not local_file.exists():
        raise FileNotFoundError(f"Active local image does not exist: {local_file}")

    split = clean(row.get("split")) or deterministic_split(relative_path, args.split_seed, args.train_ratio, args.val_ratio)
    if split not in VALID_SPLITS:
        raise ValueError(f"Invalid split '{split}' for {relative_path}; expected {sorted(VALID_SPLITS)}")

    return {
        "image_path": f"{azureml_uri_prefix.rstrip('/')}/{relative_path}",
        "local_image_path": os.path.relpath(local_file, release_dir).replace(os.sep, "/"),
        "label": row["label"],
        "split": split,
        "season": row.get("season", ""),
        "source": row.get("source", ""),
        "collected_date": row.get("collected_date", ""),
        "label_schema_version": row.get("label_schema_version", ""),
        "annotation_batch": row.get("annotation_batch", ""),
        "exclude_reason": exclude_reason,
    }


def deterministic_split(relative_path: str, seed: str, train_ratio: float, val_ratio: float) -> str:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must satisfy: train_ratio > 0, val_ratio >= 0, train_ratio + val_ratio < 1")

    digest = hashlib.sha256(f"{seed}:{relative_path}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def normalize_relative_path(value: str) -> str:
    if not value:
        raise ValueError("relative_path cannot be empty")
    path = value.strip().replace("\\", "/").lstrip("/")
    if ".." in Path(path).parts:
        raise ValueError(f"relative_path must not contain '..': {value}")
    return path


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


if __name__ == "__main__":
    main()