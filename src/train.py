"""
train.py  (v2 — time-series features)
======================================
Trains a RandomForest on FIFA historical results using engineered
time-series features from feature_engineering.py.

Run order:
    python src/feature_engineering.py   # build enriched dataset first
    python src/train.py                 # then train
"""

import argparse
import os

import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ─── Feature columns ──────────────────────────────────────────────────────────
# Original encoded features (kept for backwards compat with predict.py)
BASE_FEATURES = [
    "home_team_encoded",
    "away_team_encoded",
    "tournament_encoded",
    "neutral",
]

# New time-series features from feature_engineering.py
TS_FEATURES = [
    "home_win_rate",
    "home_draw_rate",
    "home_loss_rate",
    "home_goals_scored_avg",
    "home_goals_conceded_avg",
    "home_home_win_rate",
    "home_home_goals_avg",
    "home_streak",
    "away_win_rate",
    "away_draw_rate",
    "away_loss_rate",
    "away_goals_scored_avg",
    "away_goals_conceded_avg",
    "away_away_win_rate",
    "away_away_goals_avg",
    "away_streak",
    "h2h_home_win_rate",
    "win_rate_diff",
    "goals_diff",
    "streak_diff",
]

ALL_FEATURES = BASE_FEATURES + TS_FEATURES


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_result(row) -> str:
    if row["home_score"] > row["away_score"]:
        return "Home Win"
    elif row["home_score"] < row["away_score"]:
        return "Away Win"
    return "Draw"


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",           default="data/results_engineered.csv",
                   help="Engineered CSV from feature_engineering.py. "
                        "Falls back to data/results.csv if not found.")
    p.add_argument("--output",         default="models/fifa_model.pkl")
    p.add_argument("--encoders",       default="models/encoders.pkl")
    p.add_argument("--test-snap",      default="data/test_snapshot.csv")
    p.add_argument("--test-size",      type=float, default=0.2)
    p.add_argument("--n-estimators",   type=int,   default=200)
    p.add_argument("--random-state",   type=int,   default=42)
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    data_path = args.data
    if not os.path.exists(data_path):
        print(f"  ⚠️  '{data_path}' not found, falling back to data/results.csv")
        print("     Run feature_engineering.py first for better accuracy.\n")
        data_path = "data/results.csv"

    print(f"\n[1/6] Loading data from '{data_path}' …")
    df = pd.read_csv(data_path)
    print(f"  Shape : {df.shape}")

    has_ts = all(c in df.columns for c in TS_FEATURES)
    if has_ts:
        print(f"  ✅  Time-series features detected ({len(TS_FEATURES)} columns)")
    else:
        print("  ⚠️   Time-series features not found — using base features only")

    # ── 2. Target ─────────────────────────────────────────────────────────────
    print("\n[2/6] Creating target column …")
    df["result"] = df.apply(get_result, axis=1)
    print(df["result"].value_counts().to_string())

    # ── 3. Encode ─────────────────────────────────────────────────────────────
    print("\n[3/6] Encoding categorical features …")
    home_enc  = LabelEncoder()
    away_enc  = LabelEncoder()
    tourn_enc = LabelEncoder()

    df["home_team_encoded"]  = home_enc.fit_transform(df["home_team"])
    df["away_team_encoded"]  = away_enc.fit_transform(df["away_team"])
    df["tournament_encoded"] = tourn_enc.fit_transform(df["tournament"])

    encoders = {
        "home_team":  home_enc,
        "away_team":  away_enc,
        "tournament": tourn_enc,
    }

    # ── 4. Split ──────────────────────────────────────────────────────────────
    print("\n[4/6] Splitting data …")
    feature_cols = ALL_FEATURES if has_ts else BASE_FEATURES
    # Only use columns that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"  Using {len(feature_cols)} features")

    X = df[feature_cols]
    y = df["result"]

    # For time-series data, split by time (no shuffling) to simulate production
    if has_ts and "date" in df.columns:
        split_idx = int(len(df) * (1 - args.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        print(f"  Time-based split: train={len(X_train)}  test={len(X_test)}")
        test_df = df.iloc[split_idx:].copy()
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )
        print(f"  Random split: train={len(X_train)}  test={len(X_test)}")
        test_df = df.loc[X_test.index].copy()

    # Save test snapshot
    os.makedirs(os.path.dirname(args.test_snap) or ".", exist_ok=True)
    test_df.to_csv(args.test_snap, index=False)
    print(f"  Test snapshot saved → {args.test_snap}")

    # ── 5. Train ──────────────────────────────────────────────────────────────
    # ── Train with LightGBM (selected via PyCaret AutoML) ────────────────────
    # AutoML ran via automl.py and selected LightGBM as the best classifier.
    print(f"\n[5/6] Training LightGBM classifier (AutoML winner) …")
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split as tts
    from sklearn.preprocessing import LabelEncoder as LE

    # Encode target as integers — LightGBM requires numeric labels
    le       = LE()
    y_tr_enc = le.fit_transform(y_train)
    y_te_enc = le.transform(y_test)
    classes  = list(le.classes_)
    draw_idx = classes.index("Draw")

    model = lgb.LGBMClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
        verbose=-1,
        num_leaves=31,
        learning_rate=0.1,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    # Find best draw threshold on a validation split
    X_tr, X_val, y_tr, y_val = tts(
        X_train, y_tr_enc,
        test_size=0.15, random_state=args.random_state
    )
    model.fit(X_tr, y_tr)

    def predict_with_threshold(proba, thresh):
        preds = []
        for row in proba:
            if row[draw_idx] >= thresh:
                preds.append("Draw")
            else:
                idx = max((v, i) for i, v in enumerate(row) if i != draw_idx)[1]
                preds.append(classes[idx])
        return preds

    best_f1, best_thresh = 0, 0.33
    for thresh in [i / 100 for i in range(15, 45)]:
        val_preds  = predict_with_threshold(model.predict_proba(X_val), thresh)
        val_labels = le.inverse_transform(y_val)
        f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    print(f"  Best draw threshold: {best_thresh:.2f}  (val macro-F1={best_f1:.4f})")
    joblib.dump(best_thresh, "models/draw_threshold.pkl")

    # Retrain on full training set
    model.fit(X_train, y_tr_enc)

    # Store string class labels as custom attribute for downstream scripts
    model.str_classes_ = le.classes_

    preds = predict_with_threshold(model.predict_proba(X_test), best_thresh)
    acc   = accuracy_score(y_test, preds)
    print(f"\n  Accuracy : {acc:.4f}")
    print(classification_report(y_test, preds))

    # Feature importance (top 10)
    if has_ts:
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        print("  Top 10 most important features:")
        for feat, imp in importances.nlargest(10).items():
            print(f"    {feat:<35s}  {imp:.4f}")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    print("\n[6/6] Saving artefacts …")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joblib.dump(model,    args.output)
    joblib.dump(encoders, args.encoders)
    joblib.dump(le,       "models/target_encoder.pkl")  # decode LightGBM numeric preds

    # Save feature column list so predict.py knows what to pass in
    joblib.dump(feature_cols, "models/feature_cols.pkl")

    print(f"  Model        → {args.output}")
    print(f"  Encoders     → {args.encoders}")
    print(f"  Feature list → models/feature_cols.pkl")
    print("\n✅  Training complete.")


if __name__ == "__main__":
    main()
