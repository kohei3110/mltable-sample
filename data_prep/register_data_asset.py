from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register an Azure ML data asset YAML with Azure ML CLI v2.")
    parser.add_argument("--file", required=True, help="Data asset YAML file.")
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace-name", required=True)
    parser.add_argument("--execute", action="store_true", help="Run the Azure CLI command. Without this, print it only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yaml_file = Path(args.file)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Data asset YAML not found: {yaml_file}")

    command = [
        "az",
        "ml",
        "data",
        "create",
        "--file",
        str(yaml_file),
        "--resource-group",
        args.resource_group,
        "--workspace-name",
        args.workspace_name,
    ]

    if not args.execute:
        print("Dry run. Command to execute:")
        print(" ".join(command))
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()