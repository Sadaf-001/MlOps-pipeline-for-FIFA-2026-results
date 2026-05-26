"""
fine_tune.py  (v2 — supports time-series features)
===================================================
Fine-tune a previously saved FIFA match prediction model on new data.

Usage:
    python src/fine_tune.py \
        --new-data  data/results_engineered.csv \
        --base-model models/fifa_model.pkl \
        --encoders   models/encoders.pkl \
        --output     models/fifa_model_finetuned.pkl
"""

import argparse
import json
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_result(row) -> str:
    if row["home_score"] > row["away_score"]:
        return "Home Win"
    elif row["home_score"] < row["away_score"]:
        return "Away Win"
    return "Draw"


def load_encoders(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Encoders not found at '{path}'. Run train.py first.")
    return joblib.load(path)


def encode_new_data(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col, key in [("home_team", "home_team"), ("away_team", "away_team"),
                     ("tournament", "tournament")]:
        enc   = encoders[key]
        known = set(enc.classes_)
        df[f"{col}_encoded"] = df[col].apply(
            lambda v: enc.transform([v])[0] if v in known else -1
        )
    return df


def prepare_features(df: pd.DataFrame, feature_cols: list) -> tuple:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing {len(missing)} expected feature columns: {missing}\n"
            f"Make sure you are passing the engineered CSV "
            f"(data/results_engineered.csv), not the raw results.csv."
        )
    return df[feature_cols], df["result"]


# ─── Core fine-tuning ─────────────────────────────────────────────────────────

def warm_start_finetune(
    base_model: RandomForestClassifier,
    X_new: pd.DataFrame,
    y_new: pd.Series,
    n_new_estimators: int = 50,
) -> RandomForestClassifier:
    base_n    = base_model.n_estimators
    new_total = base_n + n_new_estimators

    params = base_model.get_params()
    params["n_estimators"] = base_n
    params["warm_start"]   = True

    fine_tuned = RandomForestClassifier(**params)
    fine_tuned.estimators_    = base_model.estimators_
    fine_tuned.classes_       = base_model.classes_
    fine_tuned.n_classes_     = base_model.n_classes_
    fine_tuned.n_outputs_     = base_model.n_outputs_
    fine_tuned.n_features_in_ = base_model.n_features_in_

    fine_tuned.set_params(n_estimators=new_total)
    fine_tuned.fit(X_new, y_new)

    print(f"  Base trees : {base_n}\n  New trees  : {n_new_estimators}\n  Total: {new_total}")
    return fine_tuned


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, label: str,
             draw_threshold: float = None) -> dict:
    if draw_threshold is not None:
        classes  = list(model.classes_)
        draw_idx = classes.index("Draw")
        proba    = model.predict_proba(X_test)
        preds    = []
        for row in proba:
            if row[draw_idx] >= draw_threshold:
                preds.append("Draw")
            else:
                idx = max((v, i) for i, v in enumerate(row) if i != draw_idx)[1]
                preds.append(classes[idx])
    else:
        preds = model.predict(X_test)

    acc    = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    print(f"\n── {label} ──")
    print(f"  Accuracy : {acc:.4f}")
    print(classification_report(y_test, preds))
    return {"accuracy": acc, "report": report}


def log_run(log_path, base_acc, ft_acc, n_new_rows, n_new_estimators, output_path):
    entry = {
        "timestamp":         datetime.utcnow().isoformat(),
        "n_new_rows":        n_new_rows,
        "n_new_estimators":  n_new_estimators,
        "base_accuracy":     round(base_acc, 4),
        "finetuned_accuracy":round(ft_acc,   4),
        "delta":             round(ft_acc - base_acc, 4),
        "output_model":      output_path,
    }
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    history = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            history = json.load(f)
    history.append(entry)
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Run logged → {log_path}")
    print(f"  Δ accuracy : {entry['delta']:+.4f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--new-data",     default="data/results_engineered.csv")
    p.add_argument("--base-model",   default="models/fifa_model.pkl")
    p.add_argument("--encoders",     default="models/encoders.pkl")
    p.add_argument("--feature-cols", default="models/feature_cols.pkl")
    p.add_argument("--draw-thresh",  default="models/draw_threshold.pkl")
    p.add_argument("--output",       default="models/fifa_model_finetuned.pkl")
    p.add_argument("--n-estimators", type=int, default=50)
    p.add_argument("--test-size",    type=float, default=0.2)
    p.add_argument("--log",          default="logs/finetune_log.json")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Load artefacts ─────────────────────────────────────────────────────
    print(f"\n[1/5] Loading artefacts …")
    base_model   = joblib.load(args.base_model)
    encoders     = load_encoders(args.encoders)
    feature_cols = joblib.load(args.feature_cols) if os.path.exists(args.feature_cols) else None
    draw_thresh  = joblib.load(args.draw_thresh)  if os.path.exists(args.draw_thresh)  else None
    if feature_cols:
        print(f"  Feature cols loaded: {len(feature_cols)} columns")
    if draw_thresh:
        print(f"  Draw threshold loaded: {draw_thresh:.2f}")

    # ── 2. Load & prepare new data ────────────────────────────────────────────
    print(f"\n[2/5] Loading new data from '{args.new_data}' …")
    df = pd.read_csv(args.new_data)
    df["result"] = df.apply(get_result, axis=1)

    # Fix neutral column if it's still strings
    if df["neutral"].dtype == object:
        df["neutral"] = df["neutral"].str.upper().map({"TRUE": 1, "FALSE": 0})

    df = encode_new_data(df, encoders)
    print(f"  {len(df)} rows | {df['result'].value_counts().to_dict()}")

    # ── 3. Build feature matrix ───────────────────────────────────────────────
    print("\n[3/5] Preparing features …")
    if feature_cols is None:
        feature_cols = ["home_team_encoded", "away_team_encoded",
                        "tournament_encoded", "neutral"]
        print("  ⚠️  No feature_cols.pkl found — using base 4 features only")

    # Sort chronologically to avoid leakage — same as train.py
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    X, y = prepare_features(df, feature_cols)

    # Time-based split: train on oldest 80%, test on most recent 20%
    split_idx = int(len(df) * (1 - args.test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # ── 4. Baseline evaluation ────────────────────────────────────────────────
    print("\n[4/5] Evaluating base model on new test split …")
    base_metrics = evaluate(base_model, X_test, y_test,
                            "BASE MODEL", draw_threshold=draw_thresh)

    # ── 5. Fine-tune & evaluate ───────────────────────────────────────────────
    print(f"\n[5/5] Fine-tuning (+{args.n_estimators} trees via warm-start) …")
    ft_model = warm_start_finetune(base_model, X_train, y_train, args.n_estimators)
    ft_metrics = evaluate(ft_model, X_test, y_test,
                          "FINE-TUNED MODEL", draw_threshold=draw_thresh)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joblib.dump(ft_model, args.output)
    print(f"\n  Fine-tuned model saved → {args.output}")

    log_run(args.log, base_metrics["accuracy"], ft_metrics["accuracy"],
            len(df), args.n_estimators, args.output)


if __name__ == "__main__":
    main()
