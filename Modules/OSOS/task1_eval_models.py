from __future__ import annotations

import argparse
import csv
import os
import random
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from sentiment140_utils import ensure_training_csv, load_sentiment140, train_test_split_df


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def evaluate_predictions(y_true: list[int], y_pred: list[int], average: str) -> dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {"acc": acc, "precision": p, "recall": r, "f1": f1}


def eval_fasttext_pretrained(
    model_path: str,
    test_texts: list[str],
    test_labels: list[int],
    average: str,
) -> dict[str, float]:
    try:
        import fasttext
    except Exception as exc:
        raise RuntimeError("fasttext is not installed. Install with: pip install fasttext") from exc

    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(
            "fastText model not found. Provide --fasttext-model pointing to a pretrained "
            "sentiment classifier .bin/.ftz file."
        )

    model = fasttext.load_model(model_path)
    pred_labels = []
    for text in test_texts:
        labels, _ = model.predict(text)
        label = labels[0] if labels else ""
        label_lower = label.lower()
        pred = 1 if ("pos" in label_lower or "positive" in label_lower) else 0
        pred_labels.append(pred)

    return evaluate_predictions(test_labels, pred_labels, average)


def eval_transformer_pretrained(
    model_name: str,
    test_texts: list[str],
    test_labels: list[int],
    batch_size: int,
    max_length: int,
    average: str,
    device: str,
) -> dict[str, float]:
    try:
        import torch
        from transformers import AutoConfig, AutoTokenizer, pipeline
    except Exception as exc:
        raise RuntimeError("transformers/torch not installed. Install with: pip install transformers torch") from exc

    if device == "cpu":
        device_id = -1
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device_id = 0
    else:
        device_id = 0 if torch.cuda.is_available() else -1

    config = AutoConfig.from_pretrained(model_name)
    id2label = {int(k): v for k, v in config.id2label.items()} if config.id2label else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    clf = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=tokenizer,
        device=device_id,
        truncation=True,
        max_length=max_length,
    )

    pred_labels = []
    for i in range(0, len(test_texts), batch_size):
        batch = test_texts[i : i + batch_size]
        outputs = clf(batch)
        for out in outputs:
            label = out.get("label", "")
            label_lower = str(label).lower()
            mapped = None
            if label_lower in ("positive", "pos", "label_1") and (
                id2label.get(1, "").lower() == "positive" or "positive" in label_lower
            ):
                mapped = 1
            elif label_lower in ("negative", "neg", "label_0") and (
                id2label.get(0, "").lower() == "negative" or "negative" in label_lower
            ):
                mapped = 0
            elif "positive" in label_lower:
                mapped = 1
            else:
                mapped = 0
            pred_labels.append(mapped)

    return evaluate_predictions(test_labels, pred_labels, average)


def write_results_csv(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "acc", "precision", "recall", "f1"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate sentiment models on Sentiment140.")
    parser.add_argument("--base-path", default="data", help="Base path containing sentiment140.zip or CSV.")
    parser.add_argument("--output-dir", default="data", help="Directory to write outputs.")
    parser.add_argument("--train-size", type=int, default=20000, help="Number of training samples.")
    parser.add_argument("--test-size", type=int, default=1000, help="Number of test samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for transformers.")
    parser.add_argument("--max-length", type=int, default=128, help="Max token length for transformers.")
    parser.add_argument("--average", default="macro", choices=["macro", "micro", "weighted"], help="Averaging.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device selection.")
    parser.add_argument("--models", default="fasttext,distilbert,bert,roberta", help="Comma list of models.")
    parser.add_argument(
        "--fasttext-model",
        default=os.path.join("data", "fasttext_sentiment.bin"),
        help="Path to pretrained fastText sentiment model.",
    )
    parser.add_argument(
        "--distilbert-model",
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="HF model name/path for DistilBERT.",
    )
    parser.add_argument(
        "--bert-model",
        default="textattack/bert-base-uncased-SST-2",
        help="HF model name/path for BERT.",
    )
    parser.add_argument(
        "--roberta-model",
        default="cardiffnlp/twitter-roberta-base-sentiment-latest",
        help="HF model name/path for RoBERTa.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    train_csv = ensure_training_csv(args.base_path)
    df = load_sentiment140(train_csv)
    split = train_test_split_df(df, args.train_size, args.test_size, args.seed)

    model_list = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    results: list[dict[str, Any]] = []

    if "fasttext" in model_list:
        try:
            metrics = eval_fasttext_pretrained(
                args.fasttext_model,
                split.test_texts,
                split.test_labels,
                args.average,
            )
            results.append({"model": "fastText", **metrics})
        except FileNotFoundError as exc:
            print(f"Skipping fastText: {exc}")

    transformer_map = {
        "distilbert": args.distilbert_model,
        "bert": args.bert_model,
        "roberta": args.roberta_model,
    }

    for key, model_name in transformer_map.items():
        if key not in model_list:
            continue
        metrics = eval_transformer_pretrained(
            model_name,
            split.test_texts,
            split.test_labels,
            args.batch_size,
            args.max_length,
            args.average,
            args.device,
        )
        display = "DistilBERT" if key == "distilbert" else key.upper()
        results.append({"model": display, **metrics})

    out_csv = os.path.join(args.output_dir, "evaluation_results.csv")
    write_results_csv(out_csv, results)
    print(f"Wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
