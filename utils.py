"""Utility helpers for working with the dialog act datasets."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt

__all__ = [
    "look_for_multiple_dialog_acts",
    "convert_data_to_lowercase",
    "plot_dialog_act_counts",
    "remove_duplicates",
]


def look_for_multiple_dialog_acts(
    input_file: str | Path,
    dialog_acts: Optional[Iterable[str]] = None,
    encoding: str = "utf-8",
) -> int:
    """Print lines that repeat the same dialog act and return the count."""
    acts = (
        list(dialog_acts)
        if dialog_acts is not None
        else [
            "ack",
            "affirm",
            "bye",
            "confirm",
            "deny",
            "hello",
            "inform",
            "negate",
            "null",
            "repeat",
            "reqalts",
            "reqmore",
            "request",
            "restart",
            "thankyou",
        ]
    )

    print(
        "## Sentences of more than one word, which also have more than than one dialog act assigned to it ##"
    )
    count = 0
    with open(input_file, "r", encoding=encoding) as fin:
        for i, line in enumerate(fin, start=1):
            words = " ".join(line.strip().replace("\t", " ").split()).split()
            if len(words) > 2 and len(words) >= 2 and words[0] == words[1] and words[0] in acts:
                count += 1
                print(i, line.rstrip("\n"))

    print(f"\nthere were {count} instances where the dialog act appeared twice")
    return count


def convert_data_to_lowercase(
    input_file: str | Path,
    output_file: str | Path,
    encoding: str = "utf-8",
) -> None:
    """Lowercase the contents of ``input_file`` and write them to ``output_file``."""
    with open(input_file, "r", encoding=encoding) as fin:
        text_lower = fin.read().lower()

    with open(output_file, "w", encoding=encoding) as fout:
        fout.write(text_lower)

    print(f"result saved to {output_file}")


def plot_dialog_act_counts(
    dat_path: str | Path,
    save_path: Optional[str | Path] = None,
    title: str = "Dialog act frequency",
    encoding: str = "utf-8",
) -> dict[str, int]:
    """Plot label frequencies for ``dat_path`` and return the counts."""
    counts: Counter[str] = Counter()
    total_lines = 0
    skipped = 0

    with open(dat_path, "r", encoding=encoding) as fin:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            total_lines += 1
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                skipped += 1
                continue
            counts[parts[0]] += 1

    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    labels = [label for label, _ in items]
    values = [value for _, value in items]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel("Dialog act")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()

    print(f"Total non-empty lines read : {total_lines}")
    print(f"Malformed/label-only lines : {skipped}")
    print("Counts:", dict(items))

    return dict(items)


def remove_duplicates(
    input_file: str | Path,
    output_file: str | Path,
    encoding: str = "utf-8",
) -> None:
    """Remove duplicated lines while preserving order."""
    seen: set[str] = set()
    unique_lines: list[str] = []
    total_lines_in_source = 0

    with open(input_file, "r", encoding=encoding) as fin:
        for line in fin:
            total_lines_in_source += 1
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)

    with open(output_file, "w", encoding=encoding) as fout:
        fout.writelines(unique_lines)

    print(f"there are a total of {total_lines_in_source} lines")
    print(f"there are {len(unique_lines)} unique lines")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Utility helpers for dialog act datasets.")
    subparsers = parser.add_subparsers(dest="command")

    parser_lower = subparsers.add_parser(
        "lowercase", help="Lowercase the contents of a dataset file"
    )
    parser_lower.add_argument("input", type=Path, help="Source file")
    parser_lower.add_argument("output", type=Path, help="Destination file")

    parser_dupes = subparsers.add_parser(
        "dedupe", help="Remove duplicate lines from a dataset file"
    )
    parser_dupes.add_argument("input", type=Path, help="Source file")
    parser_dupes.add_argument("output", type=Path, help="Destination file")

    parser_counts = subparsers.add_parser(
        "plot", help="Plot dialog act label counts from a .dat file"
    )
    parser_counts.add_argument("input", type=Path, help="Path to the .dat file")
    parser_counts.add_argument(
        "--save",
        type=Path,
        default=None,
        help="If provided, save the plot to this location instead of displaying it",
    )
    parser_counts.add_argument(
        "--title", type=str, default="Dialog act frequency", help="Plot title"
    )

    parser_multi = subparsers.add_parser(
        "check-multi",
        help="Print rows where the same dialog act label appears twice",
    )
    parser_multi.add_argument("input", type=Path, help="Path to the .dat file")

    args = parser.parse_args()

    if args.command == "lowercase":
        convert_data_to_lowercase(args.input, args.output)
    elif args.command == "dedupe":
        remove_duplicates(args.input, args.output)
    elif args.command == "plot":
        plot_dialog_act_counts(args.input, args.save, args.title)
    elif args.command == "check-multi":
        look_for_multiple_dialog_acts(args.input)
    else:
        parser.print_help()
