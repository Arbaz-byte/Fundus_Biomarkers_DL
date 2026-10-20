"""Evaluate predictions and write ONLY aggregate metrics (no images, no patient rows).

Input: a CSV with columns label, pred (and optionally prob).
Designed so it can be run inside a protected environment and only the JSON is shared.
"""
import argparse
import json

import pandas as pd

from fundus.metrics import quadratic_weighted_kappa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default="reports/aggregate_metrics.json")
    args = ap.parse_args()

    df = pd.read_csv(args.preds)
    result = {
        "n": int(len(df)),
        "accuracy": float((df["label"] == df["pred"]).mean()),
        "qwk": float(quadratic_weighted_kappa(df["label"], df["pred"])),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(result)


if __name__ == "__main__":
    main()
