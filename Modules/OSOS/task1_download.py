"""
Task 1 (offline): Download public tweet data and format it for the pipeline.

Uses the Stanford Sentiment140 dataset (CSV with tweet text + date).
Outputs a CSV with columns: tweet, date, completion (neutral).

Example:
  python Modules/OSOS/task1_download.py --base-path data --limit 50000
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import urllib.error
import zipfile

import pandas as pd


SENTIMENT140_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
TRAIN_FILENAME = "training.1600000.processed.noemoticon.csv"


def download_zip(url: str, dest_path: str, max_retries: int = 5, resume: bool = True) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading: {url}")

    for attempt in range(1, max_retries + 1):
        try:
            # Resume if a partial file exists
            existing_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
            req = urllib.request.Request(url)
            if resume and existing_size > 0:
                req.add_header("Range", f"bytes={existing_size}-")
                mode = "ab"
                print(f"Resuming download at byte {existing_size} (attempt {attempt}/{max_retries})")
            else:
                mode = "wb"
                print(f"Starting download (attempt {attempt}/{max_retries})")

            with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, mode) as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

            # If we reached here, download succeeded
            print(f"Saved: {dest_path}")
            return
        except (urllib.error.ContentTooShortError, urllib.error.URLError, TimeoutError) as e:
            print(f"Download failed: {e}")
            if attempt == max_retries:
                raise


def extract_train_csv(zip_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        if TRAIN_FILENAME not in zf.namelist():
            raise FileNotFoundError(f"{TRAIN_FILENAME} not found in zip.")
        zf.extract(TRAIN_FILENAME, out_dir)
    return os.path.join(out_dir, TRAIN_FILENAME)


def convert_to_scrap_results(train_csv: str, out_csv: str, limit: int | None) -> None:
    # Sentiment140 columns: target, ids, date, flag, user, text
    df = pd.read_csv(
        train_csv,
        header=None,
        encoding="latin-1",
        usecols=[2, 5],
        names=["date", "tweet"],
    )
    if limit:
        df = df.head(limit)
    df["completion"] = "neutral"
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} (rows={len(df)})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download public tweet data and format it.")
    parser.add_argument("--base-path", required=True, help="Directory for downloads and outputs.")
    parser.add_argument("--out-csv", default="Scrap_Results.csv", help="Output CSV filename.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows for quick tests.")
    parser.add_argument("--url", default=SENTIMENT140_URL, help="Override dataset URL if needed.")
    parser.add_argument("--skip-download", action="store_true", help="Assume zip already downloaded.")
    args = parser.parse_args()

    base_path = args.base_path
    os.makedirs(base_path, exist_ok=True)

    zip_path = os.path.join(base_path, "sentiment140.zip")
    if not args.skip_download:
        download_zip(args.url, zip_path)
        # Validate zip; if corrupt, retry once without resume.
        if not zipfile.is_zipfile(zip_path):
            print("Downloaded file is not a valid zip. Re-downloading from scratch...")
            try:
                os.remove(zip_path)
            except OSError:
                pass
            download_zip(args.url, zip_path, resume=False)
    elif not os.path.exists(zip_path):
        print(f"Missing zip: {zip_path}", file=sys.stderr)
        return 1

    train_csv = extract_train_csv(zip_path, base_path)
    out_csv = os.path.join(base_path, args.out_csv)
    convert_to_scrap_results(train_csv, out_csv, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
