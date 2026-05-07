from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dataset_card.md and split_summary.json from manifest.csv.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-card", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--name", default="animal-image-detection")
    parser.add_argument("--version", required=True)
    parser.add_argument("--release-date", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(Path(args.manifest))
    active = [row for row in rows if not clean(row.get("exclude_reason"))]
    excluded = [row for row in rows if clean(row.get("exclude_reason"))]

    summary = {
        "data_asset_name": args.name,
        "version": args.version,
        "release_date": args.release_date,
        "active_rows": len(active),
        "excluded_rows": len(excluded),
        "splits": count_by(active, "split"),
        "labels": count_by(active, "label"),
        "seasons": count_by(active, "season"),
        "sources": count_by(active, "source"),
        "source_seasons": count_pairs(active, "source", "season"),
        "exclude_reasons": count_by(excluded, "exclude_reason"),
    }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    card_path = Path(args.dataset_card)
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(render_card(args, summary), encoding="utf-8")

    print(f"Wrote {card_path}")
    print(f"Wrote {summary_path}")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: clean(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def render_card(args: argparse.Namespace, summary: dict[str, Any]) -> str:
    source_season_rows = render_source_season_rows(summary)
    return f"""# Dataset Card: {args.name}:{args.version}

## Overview

Curated image dataset release for Azure Machine Learning training jobs.

## Dataset Version

- Data asset name: {args.name}
- Version: {args.version}
- Release date: {args.release_date}

## Data Sources

| Source | Season | Count |
|---|---|---:|
{source_season_rows}

## Splits

{render_counter_table(summary['splits'], 'Split')}

## Label Distribution

{render_counter_table(summary['labels'], 'Label')}

## Exclusions

{render_counter_table(summary['exclude_reasons'], 'Reason') if summary['exclude_reasons'] else 'No excluded rows in this release.'}

## Operational Notes

- This card is generated from `manifest.csv` and should be committed with each curated release.
- Do not mutate a release folder after registering it as an Azure ML Data asset version; create a new version instead.
"""


def render_source_season_rows(summary: dict[str, Any]) -> str:
    source_seasons = summary.get("source_seasons", {})
    rows = []
    for key, count in source_seasons.items():
        source, season = key.split("|", maxsplit=1)
        rows.append(f"| {source} | {season} | {count} |")
    return "\n".join(rows) if rows else "| n/a | n/a | 0 |"


def render_counter_table(counter: dict[str, int], label: str) -> str:
    lines = [f"| {label} | Count |", "|---|---:|"]
    for key, value in counter.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def count_by(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    return dict(sorted(Counter(clean(row.get(column)) or "<empty>" for row in rows).items()))


def count_pairs(rows: list[dict[str, str]], left: str, right: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                f"{clean(row.get(left)) or '<empty>'}|{clean(row.get(right)) or '<empty>'}"
                for row in rows
            ).items()
        )
    )


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


if __name__ == "__main__":
    main()