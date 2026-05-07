from __future__ import annotations

import argparse
from pathlib import Path


MLTABLE_TEMPLATE = """$schema: https://azuremlschemas.azureedge.net/latest/MLTable.schema.json
type: mltable

paths:
  - file: ./{manifest_name}

transformations:
  - read_delimited:
      delimiter: ','
      encoding: utf8
      header: all_files_same_headers
      empty_as_string: true
      infer_column_types: false
{filter_block}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an MLTable file for a curated release folder.")
    parser.add_argument("--release-dir", required=True, help="Curated release folder that contains manifest.csv.")
    parser.add_argument("--manifest-name", default="manifest.csv")
    parser.add_argument("--output-name", default="MLTable")
    parser.add_argument("--no-filter-excluded", action="store_true", help="Do not add exclude_reason filter.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    manifest = release_dir / args.manifest_name
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    filter_block = "" if args.no_filter_excluded else "  - filter: \"col('exclude_reason') == ''\"\n"
    output = release_dir / args.output_name
    output.write_text(
        MLTABLE_TEMPLATE.format(manifest_name=args.manifest_name, filter_block=filter_block),
        encoding="utf-8",
    )
    print(f"Wrote MLTable file to {output}")


if __name__ == "__main__":
    main()