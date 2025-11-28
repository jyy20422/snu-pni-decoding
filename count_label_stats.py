#!/usr/bin/env python3
"""
Count per-label signal counts for each CSV file and for the full dataset.

Each CSV is expected to contain one signal per row, with the behavior label in
the last column. The script walks the provided dataset directory recursively
and reports label counts for each CSV along with overall totals.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


def iter_csv_files(root: Path, pattern: str) -> Iterable[Path]:
    """Yield CSV files under root honoring the provided glob pattern."""
    if root.is_file():
        return [root]
    return sorted(root.rglob(pattern))


def count_labels(csv_path: Path, skip_header: bool) -> Counter:
    """Count how many rows belong to each label in the CSV file."""
    counts: Counter = Counter()
    with csv_path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        if skip_header:
            next(reader, None)
        for row in reader:
            if not row:
                continue
            label = row[-1].strip()
            counts[label] += 1
    return counts


def format_counts(counts: Counter) -> str:
    """Return a formatted string for label counts."""
    label_parts = [
        f"{label}: {counts[label]}"
        for label in sorted(counts.keys(), key=lambda x: (len(x), x))
    ]
    return ", ".join(label_parts) or "no signals found"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count the number of signals for each label in every CSV file "
            "and for the entire dataset."
        )
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the dataset directory or a single CSV file.",
    )
    parser.add_argument(
        "--glob",
        default="*.csv",
        help="Glob pattern (relative) used to match CSV files (default: *.csv).",
    )
    parser.add_argument(
        "--skip-header",
        action="store_true",
        help="Skip the first row of each CSV if it contains a header.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path of a JSON file to store the per-file and overall counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path: Path = args.dataset_path.expanduser().resolve()

    if not dataset_path.exists():
        raise SystemExit(f"Dataset path not found: {dataset_path}")

    csv_files = iter_csv_files(dataset_path, args.glob)
    if not csv_files:
        raise SystemExit("No CSV files found. Adjust --glob or check the dataset path.")

    totals: Counter = Counter()
    file_summaries: Dict[str, Dict[str, int]] = {}

    for csv_file in csv_files:
        counts = count_labels(csv_file, args.skip_header)
        file_summaries[str(csv_file)] = dict(counts)
        totals.update(counts)

        print(f"File: {csv_file}")
        print(f"  {format_counts(counts)}")
        print(f"  total rows: {sum(counts.values())}\n")

    print("=== Dataset totals ===")
    print(f"  {format_counts(totals)}")
    print(f"  total rows: {sum(totals.values())}")

    if args.json:
        output = {
            "dataset_path": str(dataset_path),
            "files": file_summaries,
            "totals": dict(totals),
            "total_rows": sum(totals.values()),
        }
        args.json.write_text(json.dumps(output, indent=2))
        print(f"\nSaved JSON summary to {args.json}")


if __name__ == "__main__":
    main()




