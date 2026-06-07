#!/usr/bin/env python3
"""
Обучение char n-gram LM для оценки уверенности парсера.

Использование:
    python scripts/train_ngram_lm.py                         # order=4
    python scripts/train_ngram_lm.py --order 5 --smoothing 0.5

Сохраняет: models/char_ngram.pkl + печатает пороги
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import numpy as np
from src.ngram_lm import CharNGram


def main():
    parser = argparse.ArgumentParser(description="Train char n-gram LM")
    parser.add_argument("--data", default="data/synthetic.f")
    parser.add_argument("--order", type=int, default=4, help="N-gram order (default: 4)")
    parser.add_argument("--smoothing", type=float, default=1.0, help="Add-k smoothing (default: 1.0)")
    parser.add_argument("--output", default="models/char_ngram.pkl")
    args = parser.parse_args()

    print(f"Loading data: {args.data}")
    df = pl.read_ipc(args.data)
    clean = df.filter(pl.col("noise_level") == "clean")
    if clean.height == 0:
        clean = df
    gts = clean["ground_truth"].to_list()
    print(f"  Ground truths: {len(gts)}")
    print(f"  Lengths: min={min(len(g) for g in gts)}, max={max(len(g) for g in gts)}")

    print(f"\nTraining char n-gram (order={args.order}, k={args.smoothing})...")
    lm = CharNGram(order=args.order, smoothing_k=args.smoothing)
    lm.train(gts)
    print(f"  Vocab size: {lm._vocab_size}")
    print(f"  Unique n-grams: {len(lm._ngram_counts)}")
    print(f"  Unique contexts: {len(lm._context_counts)}")

    print(f"\nComputing thresholds...")
    scores = [lm.score(gt) for gt in gts]
    scores_arr = np.array(scores)
    min_score = float(np.min(scores_arr))
    p5 = float(np.percentile(scores_arr, 5))
    p10 = float(np.percentile(scores_arr, 10))
    mean_score = float(np.mean(scores_arr))
    std_score = float(np.std(scores_arr))

    # Also compute parser output scores for realistic threshold
    print(f"  Scores on train:")
    print(f"    min={min_score:.4f}")
    print(f"    5th percentile={p5:.4f}")
    print(f"    10th percentile={p10:.4f}")
    print(f"    mean={mean_score:.4f}")
    print(f"    std={std_score:.4f}")

    # Save as extra attributes
    lm.threshold_min = min_score
    lm.threshold_p5 = p5
    lm.threshold_p10 = p10
    lm.threshold_mean = mean_score
    lm.threshold_std = std_score
    lm.n_train = len(gts)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    lm.save(args.output)
    print(f"\nSaved to {args.output}")

    print(f"\nRecommended threshold (p5): {p5:.4f}")
    print("  Usage: HYBRID_NGRAM_THRESHOLD=p5 for percentile-based")

    # Quick sanity check on known ASR errors
    print(f"\nSanity check:")
    test_cases = [
        ("25 рублей", True),
        ("1000 рублей", True),
        ("свыше 22", True),
        ("двеси 350000", False),
        ("на двесте рублей", False),
        ("тыща рублей", False),
    ]
    for text, expected in test_cases:
        s = lm.score(text)
        flag = "OK" if (s >= p5) == expected else "MISMATCH"
        status = "CORRECT" if s >= p5 else "LOW CONF"
        print(f"  [{flag:9s}] {status:8s} (score={s:.4f}) {text}")


if __name__ == "__main__":
    main()
