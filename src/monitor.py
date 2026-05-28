"""
monitor.py
==========
Model monitoring for the FIFA match prediction pipeline.

Tracks:
  • Prediction distribution drift  (chi-squared test)
  • Feature drift                  (KS test per column)
  • Rolling accuracy               (when ground-truth labels arrive)
  • Alerting thresholds            (configurable)

Usage:
    python monitor.py --reference  data/results.csv \
                      --current    data/new_results.csv \
                      --model      models/fifa_model.pkl \
                      --encoders   models/encoders.pkl \
                      --report     logs/monitoring_report.json
"""

import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report


# ─── Thresholds (edit to taste) ───────────────────────────────────────────────
THRESHOLDS = {
    "accuracy_drop": 0.05,        # alert if accuracy falls by > 5 pp vs reference
    "ks_p_value": 0.05,           # alert if KS p-value < 0.05 (feature drift)
    "chi2_p_value": 0.05,         # alert if chi2 p-value < 0.05 (pred dist drift)
    "min_samples": 30,            # minimum rows needed for drift tests
}


# ─── Data preparation (reuse from train.py logic) ─────────────────────────────

def get_result(row) -> str:
    if row["home_score"] > row["away_score"]:
        return "Home Win"
    elif row["home_score"] < row["away_score"]:
        return "Away Win"
    return "Draw"


def encode_df(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col, key in [("home_team", "home_team"), ("away_team", "away_team"),
                     ("tournament", "tournament")]:
        enc = encoders[key]
        known = set(enc.classes_)
        df[f"{col}_encoded"] = df[col].apply(
            lambda v: enc.transform([v])[0] if v in known else -1
        )
    return df


BASE_FEATURE_COLS = [
    "home_team_encoded", "away_team_encoded", "tournament_encoded", "neutral"
]
FEATURE_COLS = BASE_FEATURE_COLS  # overridden below if feature_cols.pkl exists


# ─── Drift tests ──────────────────────────────────────────────────────────────

def ks_feature_drift(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    """
    Kolmogorov-Smirnov test for each numeric feature.
    Returns per-feature statistic, p-value, and drift flag.
    """
    results = {}
    for col in FEATURE_COLS:
        if col not in ref.columns or col not in cur.columns:
            continue
        stat, p = stats.ks_2samp(ref[col].dropna(), cur[col].dropna())
        drifted = p < THRESHOLDS["ks_p_value"]
        results[col] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "drifted": drifted,
        }
    return results


def chi2_prediction_drift(ref_preds: np.ndarray, cur_preds: np.ndarray) -> dict:
    """
    Chi-squared test on the prediction label distribution.
    """
    labels = ["Away Win", "Draw", "Home Win"]
    ref_counts = np.array([np.sum(ref_preds == l) for l in labels], dtype=float)
    cur_counts = np.array([np.sum(cur_preds == l) for l in labels], dtype=float)

    # Avoid zero expected freq
    if ref_counts.sum() == 0 or cur_counts.sum() == 0:
        return {"error": "Insufficient predictions for chi2 test."}

    # Normalise reference as expected proportions
    expected = ref_counts / ref_counts.sum() * cur_counts.sum()
    chi2, p = stats.chisquare(cur_counts, f_exp=expected)

    return {
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(p), 4),
        "drifted": p < THRESHOLDS["chi2_p_value"],
        "reference_distribution": dict(zip(labels, ref_counts.tolist())),
        "current_distribution":   dict(zip(labels, cur_counts.tolist())),
    }


# ─── Accuracy monitoring ──────────────────────────────────────────────────────

def rolling_accuracy(
    model, X: pd.DataFrame, y: pd.Series, window: int = 200
) -> dict:
    """
    Compute accuracy over a rolling window to surface gradual degradation.
    """
    preds = model.predict(X)
    correct = (preds == y.values).astype(int)

    if len(correct) < window:
        window = len(correct)

    rolling = pd.Series(correct).rolling(window).mean().dropna()
    return {
        "overall_accuracy": round(float(np.mean(correct)), 4),
        "rolling_window": window,
        "rolling_mean":   round(float(rolling.mean()), 4),
        "rolling_min":    round(float(rolling.min()),  4),
        "rolling_max":    round(float(rolling.max()),  4),
        "per_class": classification_report(y, preds, output_dict=True),
    }


# ─── Alerting ─────────────────────────────────────────────────────────────────

def collect_alerts(
    feature_drift: dict,
    pred_drift: dict,
    ref_accuracy: float,
    cur_accuracy: float,
) -> list:
    alerts = []

    # Feature drift alerts
    for feat, res in feature_drift.items():
        if res.get("drifted"):
            alerts.append(
                f"⚠️  FEATURE DRIFT — '{feat}' "
                f"(KS={res['ks_statistic']}, p={res['p_value']})"
            )

    # Prediction distribution drift
    if pred_drift.get("drifted"):
        alerts.append(
            f"⚠️  PREDICTION DRIFT — chi2={pred_drift['chi2_statistic']}, "
            f"p={pred_drift['p_value']}"
        )

    # Accuracy degradation
    drop = ref_accuracy - cur_accuracy
    if drop > THRESHOLDS["accuracy_drop"]:
        alerts.append(
            f"🔴  ACCURACY DROP — reference={ref_accuracy:.4f}, "
            f"current={cur_accuracy:.4f}, drop={drop:.4f}"
        )

    if not alerts:
        alerts.append("✅  No drift or performance issues detected.")

    return alerts


# ─── Report ───────────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def save_report(path: str, report: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    history = []
    if os.path.exists(path):
        with open(path) as f:
            history = json.load(f)
    history.append(report)
    with open(path, "w") as f:
        json.dump(history, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Report saved → {path}")


def print_summary(report: dict):
    print("\n" + "═" * 60)
    print("  MONITORING SUMMARY")
    print("═" * 60)
    print(f"  Reference rows : {report['reference_rows']}")
    print(f"  Current rows   : {report['current_rows']}")
    print(f"  Ref accuracy   : {report['reference_accuracy']:.4f}")
    print(f"  Cur accuracy   : {report['current_accuracy']:.4f}")
    print("\n  Feature Drift (KS test):")
    for feat, r in report["feature_drift"].items():
        flag = "🔴" if r["drifted"] else "🟢"
        print(f"    {flag} {feat:30s}  KS={r['ks_statistic']:.4f}  p={r['p_value']:.4f}")
    print(f"\n  Prediction Drift: {'🔴 YES' if report['prediction_drift'].get('drifted') else '🟢 NO'}")
    print("\n  Alerts:")
    for a in report["alerts"]:
        print(f"    {a}")
    print("═" * 60)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Monitor FIFA model for drift and degradation.")
    p.add_argument("--reference", default="data/results.csv",
                   help="Original training data CSV (reference distribution).")
    p.add_argument("--current",   default="data/new_results.csv",
                   help="New / production data CSV (current distribution).")
    p.add_argument("--model",     default="models/fifa_model.pkl")
    p.add_argument("--encoders",  default="models/encoders.pkl")
    p.add_argument("--report",    default="logs/monitoring_report.json")
    p.add_argument("--window",    type=int, default=200,
                   help="Rolling accuracy window size.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    print("[1/5] Loading model and encoders …")
    model    = joblib.load(args.model)
    encoders = joblib.load(args.encoders)

    # Load feature cols saved by train.py / automl.py
    feat_cols_path = "models/feature_cols.pkl"
    if os.path.exists(feat_cols_path):
        global FEATURE_COLS
        FEATURE_COLS = joblib.load(feat_cols_path)
        print(f"  Feature cols loaded: {len(FEATURE_COLS)} columns")
    else:
        print("  ⚠️  feature_cols.pkl not found — using base 4 features")

    print("[2/5] Loading reference data …")
    ref_df = pd.read_csv(args.reference)
    ref_df["result"] = ref_df.apply(get_result, axis=1)
    ref_df = encode_df(ref_df, encoders)

    print("[3/5] Loading current data …")
    cur_df = pd.read_csv(args.current)
    cur_df["result"] = cur_df.apply(get_result, axis=1)
    cur_df = encode_df(cur_df, encoders)

    # ── Predict ───────────────────────────────────────────────────────────────
    print("[4/5] Running predictions …")
    X_ref = ref_df[FEATURE_COLS]
    X_cur = cur_df[FEATURE_COLS]
    y_ref = ref_df["result"]
    y_cur = cur_df["result"]

    def safe_predict(m, X):
        """Handle both sklearn and PyCaret pipeline models."""
        try:
            return m.predict(X)
        except Exception:
            # PyCaret pipeline may need the target column dropped differently
            try:
                from pycaret.classification import predict_model
                result = predict_model(m, data=X)
                return result["prediction_label"].values
            except Exception as e2:
                raise RuntimeError(f"Could not predict with model: {e2}")

    ref_preds = safe_predict(model, X_ref)
    cur_preds = safe_predict(model, X_cur)

    ref_acc = accuracy_score(y_ref, ref_preds)
    cur_acc = accuracy_score(y_cur, cur_preds)

    # ── Tests ─────────────────────────────────────────────────────────────────
    print("[5/5] Running drift tests …")
    feat_drift = ks_feature_drift(X_ref, X_cur)
    pred_drift = chi2_prediction_drift(ref_preds, cur_preds)
    roll_acc   = rolling_accuracy(model, X_cur, y_cur, window=args.window)
    alerts     = collect_alerts(feat_drift, pred_drift, ref_acc, cur_acc)

    # ── Report ────────────────────────────────────────────────────────────────
    report = {
        "timestamp":          datetime.utcnow().isoformat(),
        "reference_rows":     len(ref_df),
        "current_rows":       len(cur_df),
        "reference_accuracy": round(ref_acc, 4),
        "current_accuracy":   round(cur_acc, 4),
        "accuracy_drop":      round(ref_acc - cur_acc, 4),
        "feature_drift":      feat_drift,
        "prediction_drift":   pred_drift,
        "rolling_accuracy":   roll_acc,
        "alerts":             alerts,
    }

    save_report(args.report, report)
    print_summary(report)


if __name__ == "__main__":
    main()
