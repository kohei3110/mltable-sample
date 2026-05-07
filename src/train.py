from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {"image_path", "label", "split", "label_schema_version"}
VALID_SPLITS = {"train", "val", "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-train from one MLTable/data asset input. The goal is to "
            "validate dataset versioning and dataloader boundaries, not to train "
            "a production model."
        )
    )
    parser.add_argument("--data", required=True, help="MLTable path, mounted folder, or curated release directory.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output_dir", default="outputs/local-smoke")
    parser.add_argument("--manifest-name", default="manifest.csv")
    parser.add_argument(
        "--prefer-local-paths",
        action="store_true",
        help="Use local_image_path for local smoke tests when present.",
    )
    parser.add_argument(
        "--validate-local-images",
        action="store_true",
        help="Fail if local_image_path values cannot be resolved from the curated release directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records, source_kind, data_root = load_records(args.data, args.manifest_name)
    validate_records(records)

    active_records = [row for row in records if not clean_string(row.get("exclude_reason"))]
    excluded_records = [row for row in records if clean_string(row.get("exclude_reason"))]

    if args.validate_local_images:
        validate_local_images(active_records, data_root)

    summary = build_summary(active_records, excluded_records, source_kind)
    print_summary(summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "model.txt").open("w", encoding="utf-8") as handle:
        handle.write("dummy model artifact\n")
        handle.write(f"epochs={args.epochs}\n")
        handle.write(f"data_source={source_kind}\n")
        handle.write(f"active_rows={summary['active_rows']}\n")

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Saved dummy artifacts to: {output_dir}")


def load_records(data: str, manifest_name: str) -> tuple[list[dict[str, Any]], str, Path | None]:
    """Load rows from a local manifest first, then from MLTable if available."""

    data_path = Path(data)
    if data_path.is_file() and data_path.name == manifest_name:
        return read_manifest(data_path), "manifest-file", data_path.parent

    if data_path.is_dir():
        manifest_path = data_path / manifest_name
        if manifest_path.exists():
            return read_manifest(manifest_path), "manifest-folder", data_path

    try:
        return load_mltable_records(data), "mltable", None
    except ImportError as exc:
        raise RuntimeError(
            f"No local {manifest_name} was found under '{data}', and the mltable package is not installed. "
            "Install optional dependencies or pass a curated release folder that contains manifest.csv."
        ) from exc
    except Exception as exc:  # pragma: no cover - depends on Azure/MLTable runtime
        raise RuntimeError(
            f"Failed to load '{data}' as MLTable. For local tests, pass a folder containing {manifest_name}."
        ) from exc


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: clean_string(value) for key, value in row.items()} for row in reader]


def load_mltable_records(data: str) -> list[dict[str, Any]]:
    import mltable

    table = mltable.load(data)
    dataframe = table.to_pandas_dataframe()
    records: list[dict[str, Any]] = []
    for row in dataframe.to_dict(orient="records"):
        records.append({key: normalize_dataframe_value(value) for key, value in row.items()})
    return records


def validate_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError("Dataset manifest is empty.")

    available_columns = set(records[0])
    missing_columns = sorted(REQUIRED_COLUMNS - available_columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    invalid_splits = sorted({clean_string(row.get("split")) for row in records if clean_string(row.get("split")) not in VALID_SPLITS})
    if invalid_splits:
        raise ValueError(f"Invalid split values: {invalid_splits}. Expected one of {sorted(VALID_SPLITS)}")


def validate_local_images(records: list[dict[str, Any]], data_root: Path | None) -> None:
    if data_root is None:
        raise ValueError("--validate-local-images requires a local curated release folder input.")

    missing: list[str] = []
    for row in records:
        local_value = clean_string(row.get("local_image_path"))
        if not local_value:
            continue
        local_path = Path(local_value)
        if not local_path.is_absolute():
            local_path = data_root / local_path
        if not local_path.exists():
            missing.append(str(local_path))

    if missing:
        raise FileNotFoundError("Missing local image files:\n" + "\n".join(missing))


def build_summary(active_records: list[dict[str, Any]], excluded_records: list[dict[str, Any]], source_kind: str) -> dict[str, Any]:
    return {
        "source_kind": source_kind,
        "active_rows": len(active_records),
        "excluded_rows": len(excluded_records),
        "splits": count_by(active_records, "split"),
        "labels": count_by(active_records, "label"),
        "seasons": count_by(active_records, "season"),
        "sources": count_by(active_records, "source"),
        "label_schema_versions": count_by(active_records, "label_schema_version"),
        "exclude_reasons": count_by(excluded_records, "exclude_reason"),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Loaded from: {summary['source_kind']}")
    print(f"Active samples: {summary['active_rows']}")
    print(f"Excluded samples: {summary['excluded_rows']}")
    print(f"Splits: {summary['splits']}")
    print(f"Labels: {summary['labels']}")
    print(f"Seasons: {summary['seasons']}")
    print(f"Sources: {summary['sources']}")


def count_by(records: list[dict[str, Any]], column: str) -> dict[str, int]:
    counter = Counter(clean_string(row.get(column)) or "<empty>" for row in records)
    return dict(sorted(counter.items()))


def normalize_dataframe_value(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if value != value:  # NaN check without importing pandas.
            return ""
    except Exception:
        pass
    return value


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise