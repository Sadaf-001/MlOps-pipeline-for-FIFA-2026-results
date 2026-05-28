"""
automl.py  (v2 — time-series features)
=======================================
Uses PyCaret to automatically find the best classifier on the engineered
feature set. Saves the winning model and metadata so the rest of the
pipeline (fine_tune.py, monitor.py, test_showcase.py) can use it.

Run order:
    python src/feature_engineering.py   # build enriched dataset first
    python src/automl.py                # find best model via AutoML
    # fine_tune / monitor / test_showcase work unchanged after this

Usage:
    python src/automl.py \
        --data   data/results_engineered.csv \
        --output models/fifa_model.pkl \
        --n-top  5
"""

import argparse
import os

import joblib
import pandas as pd


# ─── Feature columns (must match feature_engineering.py output) ───────────────
BASE_FEATURES = [
    "home_team_encoded",
    "away_team_encoded",
    "tournament_encoded",
    "neutral",
]

TS_FEATURES = [
    "home_win_rate", "home_draw_rate", "home_loss_rate",
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "home_home_win_rate", "home_home_goals_avg", "home_streak",
    "away_win_rate", "away_draw_rate", "away_loss_rate",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "away_away_win_rate", "away_away_goals_avg", "away_streak",
    "h2h_home_win_rate", "win_rate_diff", "goals_diff", "streak_diff",
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
    p.add_argument("--data",    default="data/results_engineered.csv",
                   help="Engineered CSV from feature_engineering.py.")
    p.add_argument("--output",  default="models/fifa_model.pkl")
    p.add_argument("--encoders",default="models/encoders.pkl")
    p.add_argument("--n-top",   type=int, default=5,
                   help="Number of top models PyCaret compares.")
    p.add_argument("--test-size", type=float, default=0.2)
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading data from '{args.data}' …")
    df = pd.read_csv(args.data)

    has_ts = all(c in df.columns for c in TS_FEATURES)
    if not has_ts:
        print("  ⚠️  Time-series features not found — run feature_engineering.py first")

    # Fix neutral column
    if df["neutral"].dtype == object:
        df["neutral"] = df["neutral"].str.upper().map({"TRUE": 1, "FALSE": 0})

    print(f"  {len(df)} rows loaded")

    # ── 2. Target + encode ────────────────────────────────────────────────────
    print("\n[2/5] Preparing target and encoders …")
    df["result"] = df.apply(get_result, axis=1)

    from sklearn.preprocessing import LabelEncoder
    home_enc  = LabelEncoder()
    away_enc  = LabelEncoder()
    tourn_enc = LabelEncoder()
    df["home_team_encoded"]  = home_enc.fit_transform(df["home_team"])
    df["away_team_encoded"]  = away_enc.fit_transform(df["away_team"])
    df["tournament_encoded"] = tourn_enc.fit_transform(df["tournament"])

    encoders = {"home_team": home_enc, "away_team": away_enc, "tournament": tourn_enc}
    os.makedirs(os.path.dirname(args.encoders) or ".", exist_ok=True)
    joblib.dump(encoders, args.encoders)
    print(f"  Encoders saved → {args.encoders}")

    # ── 3. Time-based split ───────────────────────────────────────────────────
    print("\n[3/5] Splitting data (time-based) …")
    feature_cols = [c for c in (ALL_FEATURES if has_ts else BASE_FEATURES)
                    if c in df.columns]

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    split_idx   = int(len(df) * (1 - args.test_size))
    train_df    = df.iloc[:split_idx].copy()
    test_df     = df.iloc[split_idx:].copy()

    print(f"  Train: {len(train_df)}  |  Test: {len(test_df)}")
    print(f"  Using {len(feature_cols)} features")

    # Save test snapshot for showcase
    test_df.to_csv("data/test_snapshot.csv", index=False)
    print(f"  Test snapshot saved → data/test_snapshot.csv")

    # ── 4. PyCaret AutoML ─────────────────────────────────────────────────────
    print("\n[4/5] Running PyCaret AutoML …")
    try:
        from pycaret.classification import (
            setup, compare_models, finalize_model,
            predict_model, pull
        )

        # PyCaret setup on training data only
        train_pycaret = train_df[feature_cols + ["result"]].copy()

        setup(
            data=train_pycaret,
            target="result",
            session_id=42,
            verbose=False,
            # Class weights to handle imbalance
            fix_imbalance=True,
        )

        print(f"  Comparing top {args.n_top} models …")
        best_model = compare_models(n_select=1, verbose=True)
        results    = pull()
        print("\n  AutoML leaderboard:")
        print(results.head(args.n_top).to_string())

        # Finalize on full training data
        final_model = finalize_model(best_model)

        # Evaluate on held-out test set
        print("\n  Evaluating on held-out test set …")
        test_pycaret = test_df[feature_cols + ["result"]].copy()
        predictions  = predict_model(final_model, data=test_pycaret)
        from sklearn.metrics import accuracy_score, classification_report
        acc = accuracy_score(predictions["result"], predictions["prediction_label"])
        print(f"  Test accuracy: {acc:.4f}")
        print(classification_report(predictions["result"],
                                    predictions["prediction_label"]))

        best = final_model

    except ImportError:
        print("  ⚠️  PyCaret not installed. Falling back to RandomForest.")
        print("       Install with: pip install pycaret")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, classification_report

        X_train = train_df[feature_cols]
        y_train = train_df["result"]
        X_test  = test_df[feature_cols]
        y_test  = test_df["result"]

        best = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1,
            class_weight={"Home Win": 1.0, "Away Win": 2.0, "Draw": 4.0}
        )
        best.fit(X_train, y_train)
        preds = best.predict(X_test)
        print(f"  Test accuracy: {accuracy_score(y_test, preds):.4f}")
        print(classification_report(y_test, preds))

    # ── 5. Save ───────────────────────────────────────────────────────────────
    print("\n[5/5] Saving artefacts …")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joblib.dump(best, args.output)
    joblib.dump(feature_cols, "models/feature_cols.pkl")
    print(f"  Model        → {args.output}")
    print(f"  Feature list → models/feature_cols.pkl")
    print("\n✅  AutoML complete.")


if __name__ == "__main__":
    main()
