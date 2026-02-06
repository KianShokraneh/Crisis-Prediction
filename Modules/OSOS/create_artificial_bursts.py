"""
Create artificial bursts in a tweet dataset by spreading dates across a range
and duplicating rows on selected burst days.

Example:
  python Modules/OSOS/create_artificial_bursts.py \
    --input Modules/OSOS/data/Scrap_Results.csv \
    --output Modules/OSOS/data/Scrap_Results_bursty.csv \
    --start-date 2009-04-01 \
    --days 30 \
    --burst-days 3 \
    --burst-multiplier 4 \
    --seed 42
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create artificial bursts in tweet data.")
    parser.add_argument(
        "--input",
        default="data/Scrap_Results.csv",
        help="Input CSV with columns: date, tweet, completion",
    )
    parser.add_argument(
        "--output",
        default="data/Scrap_Results_bursty.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--start-date",
        default="2009-04-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument("--days", type=int, default=365, help="Number of days to spread data over")
    parser.add_argument("--burst-days", type=int, default=3, help="Number of burst windows to create")
    parser.add_argument(
        "--burst-lengths",
        default="5,10,20",
        help="Comma-separated burst window lengths (e.g., 5,10,20)",
    )
    parser.add_argument("--burst-multiplier", type=int, default=4, help="How many times larger burst days are")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        print(f"Input file not found: {in_path}")
        return 1

    df = pd.read_csv(in_path)
    if "tweet" not in df.columns:
        print("Expected column 'tweet' in input CSV")
        return 1

    # Build date range
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    days = [start + timedelta(days=i) for i in range(args.days)]
    if args.burst_days > args.days:
        print("burst-days cannot exceed days")
        return 1

    # Choose burst day start indices from the later half to align with rising trend
    late_start = max(0, args.days // 2)
    burst_lengths = [int(x) for x in args.burst_lengths.split(",") if x.strip()]
    if not burst_lengths:
        print("burst-lengths must contain at least one length")
        return 1
    if len(burst_lengths) != args.burst_days:
        print("Number of burst lengths must match burst-days")
        return 1
    max_start = max(late_start, args.days - max(burst_lengths))
    burst_start_indices = set(random.sample(range(late_start, max_start + 1), args.burst_days))
    burst_day_indices = set()
    for start_idx, length in zip(sorted(burst_start_indices), burst_lengths):
        for offset in range(length):
            idx = start_idx + offset
            if idx < args.days:
                burst_day_indices.add(idx)

    # Assign each original row to a day using a smooth upward trend.
    # Later days get higher probability.
    df = df.copy()
    weights = [i + 1 for i in range(args.days)]  # linear upward trend
    total = sum(weights)
    probs = [w / total for w in weights]
    chosen_days = random.choices(days, weights=probs, k=len(df))
    df["date"] = [d.isoformat() for d in chosen_days]

    # Create bursts by duplicating rows for burst days
    burst_rows = []
    for idx in burst_day_indices:
        day = days[idx].isoformat()
        day_rows = df[df["date"] == day]
        if day_rows.empty:
            continue
        # duplicate rows (multiplier-1 additional copies)
        for _ in range(args.burst_multiplier - 1):
            burst_rows.append(day_rows.sample(n=len(day_rows), replace=True, random_state=random.randint(0, 10_000)))

    if burst_rows:
        df = pd.concat([df] + burst_rows, ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(f"Rows: {len(df)}")
    print(f"Unique days: {df['date'].nunique()}")
    print(
        f"Burst windows: {args.burst_days} (lengths {burst_lengths}) "
        f"(multiplier {args.burst_multiplier})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
