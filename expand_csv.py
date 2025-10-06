#!/usr/bin/env python3
"""Augment a restaurant info CSV with simulated experience columns."""
import argparse
import csv
import random
from pathlib import Path

food_quality_options = ("mediocre", "good", "great")
crowdedness_options = ("low", "medium", "busy")
length_of_stay_options = ("short", "medium", "long")


def expand_csv(input_csv: Path, output_csv: Path) -> None:
    """Augment a restaurant CSV with simulated experience fields and save it.

    Inputs:
        input_csv: Path to the source CSV containing baseline restaurant metadata.
        output_csv: Destination path where the enriched CSV will be written.
    Returns:
        None; writes out the augmented dataset unless the output already exists.
    """
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_csv}")

    if output_csv.exists():
        # Skip writing when the target already exists to honor the no-overwrite requirement
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with input_csv.open("r", newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        base_fields = reader.fieldnames or []
        new_fields = [
            "food_quality",
            "crowdedness",
            "length_of_stay",
        ]
        fieldnames = base_fields + [field for field in new_fields if field not in base_fields]

        rows = []
        for row in reader:
            row.setdefault("food_quality", random.choice(food_quality_options))
            row.setdefault("crowdedness", random.choice(crowdedness_options))
            row.setdefault("length_of_stay", random.choice(length_of_stay_options))
            rows.append(row)

    with output_csv.open("w", newline="", encoding="utf-8") as dest:
        writer = csv.DictWriter(dest, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Configure CLI flags for augmenting restaurant metadata CSV files.

    Inputs:
        None directly; reads arguments from `sys.argv`.
    Returns:
        argparse.Namespace capturing `input_csv` and `output_csv` paths.
    """
    parser = argparse.ArgumentParser(
        description="Add synthetic experience columns to a restaurant info CSV."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the existing restaurant info CSV file",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Path to write the expanded CSV file",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for expanding a restaurant CSV via command-line usage.

    Inputs:
        None directly; relies on parsed CLI arguments.
    Returns:
        None; delegates to `expand_csv` to perform augmentation.
    """
    args = parse_args()
    expand_csv(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
