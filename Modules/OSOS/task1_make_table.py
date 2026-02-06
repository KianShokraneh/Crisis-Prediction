from __future__ import annotations

import argparse
import csv
import os


def format_float(value: str) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Render evaluation CSV as a Markdown table.")
    parser.add_argument("--input", default=os.path.join("data", "evaluation_results.csv"))
    parser.add_argument("--output", default=os.path.join("data", "evaluation_table.md"))
    args = parser.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    headers = ["Model", "Acc", "P", "R", "F1"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("model", ""),
                    format_float(row.get("acc", "")),
                    format_float(row.get("precision", "")),
                    format_float(row.get("recall", "")),
                    format_float(row.get("f1", "")),
                ]
            )
            + " |"
        )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
