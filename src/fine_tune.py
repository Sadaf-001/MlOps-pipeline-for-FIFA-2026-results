"""
fine_tune.py  (v5 - Ridge + MLflow)
=====================================
Fine-tune the FIFA match prediction model on new data.
Handles both Ridge (decision_function) and LightGBM (predict_proba).

Usage:
    python src/fine_tune.py \
        --new-data  data/results_engineered.csv \
        --base-model models/fifa_model.pkl \
        --encoders   models/encoders.pkl
"""

import argparse
import json
import os
from datetime import datetime

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


# ---- Helpers -----------------------------------------------------------------

def get_result(row) -> str:
    if row["home_score"] > row["away_score"]:
        return "Home Win"
    elif row["home_score"] < row["away_score"]:
        return "Away Win"
    return "Draw"


def load_encoders(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Encoders not found at '{path}'. Run train.py first.")
    return joblib.load(path)


def encode_new_data(df, encoders):
    df = df.copy()
    for col, key in [("home_team", "home_team"), ("away_team", "away_team"),
                     ("tournament", "tournament")]:
        enc   = encoders[key]
        known = set(enc.classes_)
        df[f"{col}_encoded"] = df[col].apply(
            lambda v: enc.transform([v])[0] if v in known else -1
        )
    return df


def get_classes(model):
    """Get string class labels for any model type."""
    if hasattr(model, 'str_classes_'):
        return list(model.str_classes_)
    classes = list(model.classes_)
    if all(isinstance(c, (int, float)) for c in classes):
        return ["Away Win", "Draw", "Home Win"]
    return [str(c) for c in classes]


def predict_with_threshold(model, X, thresh):
    """Works for both Ridge (decision_function) and LightGBM (predict_proba)."""
    classes  = get_classes(model)
    draw_idx = classes.index("Draw")

    # Ridge uses decision_function; others use predict_proba
    if hasattr(model, 'decision_function') and not hasattr(model, 'predict_proba'):
        scores = model.decision_function(X)
    elif hasattr(model, 'predict_proba'):
        scores = model.predict_proba(X)
        # For LightGBM with numeric classes, map back to strings
        raw_classes = list(model.classes_)
        if all(isinstance(c, (int, float)) for c in raw_classes):
            # scores already in right order, just need string classes
            pass
    else:
        return [str(p) for p in model.predict(X)]

    preds = []
    for row in scores:
        if thresh is not None and row[draw_idx] >= thresh:
            preds.append("Draw")
        else:
            idx = max((v, i) for i, v in enumerate(row) if i != draw_idx)[1]
            preds.append(classes[idx])
    return preds


def evaluate(model, X_test, y_test, label, draw_threshold=None):
    preds  = predict_with_threshold(model, X_test, draw_threshold)
    y_str  = [str(v) for v in y_test]
    acc    = accuracy_score(y_str, preds)
    mf1    = f1_score(y_str, preds, average="macro", zero_division=0)
    report = classification_report(y_str, preds, output_dict=True, zero_division=0)
    print(f"\n-- {label} --")
    print(f"  Accuracy : {acc:.4f}  |  Macro F1 : {mf1:.4f}")
    print(classification_report(y_str, preds, zero_division=0))
    return {"accuracy": acc, "macro_f1": mf1, "report": report}


def log_run(log_path, base_acc, ft_acc, base_f1, ft_f1, n_new_rows, output_path):
    entry = {
        "timestamp":          datetime.utcnow().isoformat(),
        "n_new_rows":         n_new_rows,
        "base_accuracy":      round(base_acc, 4),
        "finetuned_accuracy": round(ft_acc,   4),
        "acc_delta":          round(ft_acc - base_acc, 4),
        "base_macro_f1":      round(base_f1,  4),
        "finetuned_macro_f1": round(ft_f1,    4),
        "f1_delta":           round(ft_f1 - base_f1, 4),
        "output_model":       output_path,
    }
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    history = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            history = json.load(f)
    history.append(entry)
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Run logged -> {log_path}")
    print(f"  Acc delta  : {entry['acc_delta']:+.4f}")
    print(f"  F1 delta   : {entry['f1_delta']:+.4f}")


# ---- CLI ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--new-data",     default="data/results_engineered.csv")
    p.add_argument("--base-model",   default="models/fifa_model.pkl")
    p.add_argument("--encoders",     default="models/encoders.pkl")
    p.add_argument("--feature-cols", default="models/feature_cols.pkl")
    p.add_argument("--draw-thresh",  default="models/draw_threshold.pkl")
    p.add_argument("--output",       default="models/fifa_model_finetuned.pkl")
    p.add_argument("--test-size",    type=float, default=0.2)
    p.add_argument("--log",          default="logs/finetune_log.json")
    p.add_argument("--experiment",   default="FIFA Match Prediction")
    return p.parse_args()


def main():
    args = parse_args()

    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="fine_tune"):

        print(f"\n[1/5] Loading artefacts ...")
        base_model   = joblib.load(args.base_model)
        encoders     = load_encoders(args.encoders)
        feature_cols = joblib.load(args.feature_cols) if os.path.exists(args.feature_cols) else None
        draw_thresh  = joblib.load(args.draw_thresh)  if os.path.exists(args.draw_thresh)  else None
        model_type   = joblib.load("models/model_type.pkl") if os.path.exists("models/model_type.pkl") else "unknown"
        print(f"  Model type    : {model_type}")
        if feature_cols:
            print(f"  Feature cols  : {len(feature_cols)} columns")
        if draw_thresh is not None:
            print(f"  Draw threshold: {draw_thresh}")

        print(f"\n[2/5] Loading new data from '{args.new_data}' ...")
        df = pd.read_csv(args.new_data)
        df["result"] = df.apply(get_result, axis=1)
        if df["neutral"].dtype == object:
            df["neutral"] = df["neutral"].str.upper().map({"TRUE": 1, "FALSE": 0})
        df = encode_new_data(df, encoders)
        print(f"  {len(df)} rows | {df['result'].value_counts().to_dict()}")

        print("\n[3/5] Preparing features ...")
        if feature_cols is None:
            feature_cols = ["home_team_encoded", "away_team_encoded",
                            "tournament_encoded", "neutral"]
        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)

        X = df[feature_cols]
        y = df["result"]
        split_idx = int(len(df) * (1 - args.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print("\n[4/5] Evaluating base model ...")
        base_metrics = evaluate(base_model, X_test, y_test,
                                "BASE MODEL", draw_threshold=draw_thresh)

        print(f"\n[5/5] Fine-tuning ...")
        import copy
        ft_model = copy.deepcopy(base_model)
        ft_model.fit(X_train, y_train)
        ft_metrics = evaluate(ft_model, X_test, y_test,
                              "FINE-TUNED MODEL", draw_threshold=draw_thresh)

        mlflow.log_params({
            "model_type":    model_type,
            "n_new_rows":    len(df),
            "draw_threshold": draw_thresh,
        })
        mlflow.log_metrics({
            "base_accuracy":      round(base_metrics["accuracy"], 4),
            "finetuned_accuracy": round(ft_metrics["accuracy"],   4),
            "acc_delta":          round(ft_metrics["accuracy"] - base_metrics["accuracy"], 4),
            "base_macro_f1":      round(base_metrics["macro_f1"], 4),
            "finetuned_macro_f1": round(ft_metrics["macro_f1"],   4),
            "f1_delta":           round(ft_metrics["macro_f1"] - base_metrics["macro_f1"], 4),
        })

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        joblib.dump(ft_model, args.output)
        mlflow.log_artifact(args.output)
        print(f"\n  Fine-tuned model saved -> {args.output}")

        log_run(args.log,
                base_metrics["accuracy"], ft_metrics["accuracy"],
                base_metrics["macro_f1"], ft_metrics["macro_f1"],
                len(df), args.output)


if __name__ == "__main__":
    main()
