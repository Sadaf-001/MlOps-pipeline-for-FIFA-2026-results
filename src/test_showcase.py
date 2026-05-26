"""
test_showcase.py
================
Side-by-side comparison of base model vs fine-tuned model on the held-out test set.

Outputs:
  • Console table with per-row predictions (sample)
  • Per-class metrics comparison
  • Confusion matrix for each model
  • JSON report saved to logs/showcase_report.json

Usage:
    python test_showcase.py \
        --test-snap   data/test_snapshot.csv \
        --base-model  models/fifa_model.pkl \
        --ft-model    models/fifa_model_finetuned.pkl \
        --encoders    models/encoders.pkl \
        --n-sample    20
"""

import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "home_team_encoded", "away_team_encoded", "tournament_encoded", "neutral"
]
LABELS = ["Away Win", "Draw", "Home Win"]


def fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_confusion(cm: np.ndarray, model_name: str):
    print(f"\n  Confusion Matrix — {model_name}")
    header = f"{'':>12}" + "".join(f"{l:>12}" for l in LABELS)
    print(f"  {header}")
    for i, row_label in enumerate(LABELS):
        row = f"  {row_label:>12}" + "".join(f"{cm[i, j]:>12}" for j in range(len(LABELS)))
        print(row)


def per_class_table(base_rep: dict, ft_rep: dict) -> str:
    lines = [
        f"\n  {'Class':<12}  {'Base P':>8}  {'FT P':>8}  {'Base R':>8}  "
        f"{'FT R':>8}  {'Base F1':>8}  {'FT F1':>8}",
        "  " + "─" * 72,
    ]
    for cls in LABELS:
        b = base_rep.get(cls, {})
        f = ft_rep.get(cls, {})
        lines.append(
            f"  {cls:<12}  "
            f"{fmt_pct(b.get('precision', 0)):>8}  {fmt_pct(f.get('precision', 0)):>8}  "
            f"{fmt_pct(b.get('recall', 0)):>8}  {fmt_pct(f.get('recall', 0)):>8}  "
            f"{fmt_pct(b.get('f1-score', 0)):>8}  {fmt_pct(f.get('f1-score', 0)):>8}"
        )
    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-snap",  default="data/test_snapshot.csv")
    p.add_argument("--base-model", default="models/fifa_model.pkl")
    p.add_argument("--ft-model",   default="models/fifa_model_finetuned.pkl")
    p.add_argument("--encoders",   default="models/encoders.pkl")
    p.add_argument("--feature-cols", default="models/feature_cols.pkl")
    p.add_argument("--draw-thresh",  default="models/draw_threshold.pkl")
    p.add_argument("--n-sample",   type=int, default=20,
                   help="Number of sample rows to display in the per-match table.")
    p.add_argument("--report",     default="logs/showcase_report.json")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading artefacts …")
    base_model   = joblib.load(args.base_model)
    encoders     = joblib.load(args.encoders)
    feature_cols = joblib.load(args.feature_cols) if os.path.exists(args.feature_cols) else None
    draw_thresh  = joblib.load(args.draw_thresh)  if os.path.exists(args.draw_thresh)  else None
    if feature_cols:
        print(f"  Feature cols loaded: {len(feature_cols)} columns")
    if draw_thresh:
        print(f"  Draw threshold loaded: {draw_thresh:.2f}")

    ft_available = os.path.exists(args.ft_model)
    if ft_available:
        ft_model = joblib.load(args.ft_model)
        print(f"  ✅  Fine-tuned model loaded from '{args.ft_model}'")
    else:
        ft_model = None
        print(f"  ⚠️   Fine-tuned model not found at '{args.ft_model}'. "
              "Run fine_tune.py first for a full comparison.")

    print("\n[2/4] Loading test snapshot …")
    df = pd.read_csv(args.test_snap)
    cols = feature_cols if feature_cols else FEATURE_COLS
    # Only keep cols that exist in the snapshot
    cols = [c for c in cols if c in df.columns]
    X  = df[cols]
    y  = df["result"]
    print(f"  {len(df)} test rows | classes: {y.value_counts().to_dict()}")

    # ── Predictions ───────────────────────────────────────────────────────────
    print("\n[3/4] Running predictions …")
    def predict(model, X, thresh):
        if thresh is None:
            return model.predict(X)
        classes  = list(model.classes_)
        draw_idx = classes.index("Draw")
        proba    = model.predict_proba(X)
        preds    = []
        for row in proba:
            if row[draw_idx] >= thresh:
                preds.append("Draw")
            else:
                idx = max((v, i) for i, v in enumerate(row) if i != draw_idx)[1]
                preds.append(classes[idx])
        return preds

    base_preds = predict(base_model, X, draw_thresh)
    base_acc   = accuracy_score(y, base_preds)
    base_rep   = classification_report(y, base_preds, output_dict=True, zero_division=0)
    base_cm    = confusion_matrix(y, base_preds, labels=LABELS)

    if ft_model:
        ft_preds = predict(ft_model, X, draw_thresh)
        ft_acc   = accuracy_score(y, ft_preds)
        ft_rep   = classification_report(y, ft_preds, output_dict=True, zero_division=0)
        ft_cm    = confusion_matrix(y, ft_preds, labels=LABELS)
    else:
        ft_preds = np.full(len(y), "N/A")
        ft_acc   = None
        ft_rep   = {}
        ft_cm    = None

    # ── Per-match sample table ─────────────────────────────────────────────────
    print("\n[4/4] Building showcase …")

    sample = df[["home_team", "away_team", "result"]].copy().head(args.n_sample)
    sample["base_pred"] = base_preds[: args.n_sample]
    sample["base_correct"] = sample["result"] == sample["base_pred"]
    if ft_model:
        sample["ft_pred"] = ft_preds[: args.n_sample]
        sample["ft_correct"] = sample["result"] == sample["ft_pred"]

    print("\n" + "═" * 80)
    print("  PER-MATCH SHOWCASE (first {} rows)".format(args.n_sample))
    print("═" * 80)
    col_w = [20, 20, 12, 14, 10]
    hdr_cols = ["Home Team", "Away Team", "True Result", "Base Pred", "Base ✓"]
    if ft_model:
        col_w += [14, 10]
        hdr_cols += ["FT Pred", "FT ✓"]
    header = "  " + "".join(f"{h:<{w}}" for h, w in zip(hdr_cols, col_w))
    print(header)
    print("  " + "─" * (sum(col_w)))

    for _, row in sample.iterrows():
        vals = [
            str(row["home_team"])[:18],
            str(row["away_team"])[:18],
            row["result"],
            row["base_pred"],
            "✓" if row["base_correct"] else "✗",
        ]
        if ft_model:
            vals += [row["ft_pred"], "✓" if row["ft_correct"] else "✗"]
        print("  " + "".join(f"{v:<{w}}" for v, w in zip(vals, col_w)))

    # ── Accuracy comparison ────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  OVERALL ACCURACY")
    print("═" * 80)
    print(f"  Base model      : {fmt_pct(base_acc)}")
    if ft_model:
        delta = ft_acc - base_acc
        sign  = "+" if delta >= 0 else ""
        print(f"  Fine-tuned model: {fmt_pct(ft_acc)}   ({sign}{fmt_pct(delta)})")

    # ── Per-class comparison ──────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  PER-CLASS METRICS (Precision / Recall / F1)")
    print("═" * 80)
    print(per_class_table(base_rep, ft_rep if ft_model else {}))

    # ── Confusion matrices ────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  CONFUSION MATRICES")
    print("═" * 80)
    print_confusion(base_cm, "Base Model")
    if ft_cm is not None:
        print_confusion(ft_cm, "Fine-Tuned Model")

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "timestamp":    datetime.utcnow().isoformat(),
        "n_test_rows":  len(df),
        "base_accuracy": round(base_acc, 4),
        "ft_accuracy":   round(ft_acc, 4) if ft_acc is not None else None,
        "delta":         round(ft_acc - base_acc, 4) if ft_acc is not None else None,
        "base_per_class": {
            cls: {k: round(v, 4) for k, v in base_rep[cls].items()}
            for cls in LABELS if cls in base_rep
        },
        "ft_per_class": {
            cls: {k: round(v, 4) for k, v in ft_rep[cls].items()}
            for cls in LABELS if cls in ft_rep
        } if ft_model else {},
        "base_confusion_matrix": base_cm.tolist(),
        "ft_confusion_matrix":   ft_cm.tolist() if ft_cm is not None else None,
    }

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {args.report}")
    print("\n✅  Showcase complete.")


if __name__ == "__main__":
    main()
