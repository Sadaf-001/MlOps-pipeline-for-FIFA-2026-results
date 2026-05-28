"""
train.py  (v5 - LightGBM + MLflow tracking)
=============================================
Trains a LightGBM classifier on FIFA historical results using engineered
time-series features. LightGBM was selected by PyCaret AutoML.

Run order:
    python src/feature_engineering.py
    python src/train.py

MLflow UI:
    mlflow ui
    open http://localhost:5000
"""

import argparse
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ---- Feature columns ---------------------------------------------------------
BASE_FEATURES = [
    "home_team_encoded", "away_team_encoded", "tournament_encoded", "neutral",
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


# ---- Helpers -----------------------------------------------------------------

def get_result(row) -> str:
    if row["home_score"] > row["away_score"]:
        return "Home Win"
    elif row["home_score"] < row["away_score"]:
        return "Away Win"
    return "Draw"


# ---- CLI ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",           default="data/results_engineered.csv")
    p.add_argument("--output",         default="models/fifa_model.pkl")
    p.add_argument("--encoders",       default="models/encoders.pkl")
    p.add_argument("--test-snap",      default="data/test_snapshot.csv")
    p.add_argument("--test-size",      type=float, default=0.2)
    p.add_argument("--n-estimators",   type=int,   default=200)
    p.add_argument("--random-state",   type=int,   default=42)
    p.add_argument("--experiment",     default="FIFA Match Prediction")
    return p.parse_args()


# ---- Main --------------------------------------------------------------------

def main():
    args = parse_args()

    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="lgbm_train"):

        # 1. Load
        print(f"\n[1/6] Loading data from '{args.data}' ...")
        data_path = args.data if os.path.exists(args.data) else "data/results.csv"
        df = pd.read_csv(data_path)
        print(f"  Shape : {df.shape}")

        has_ts = all(c in df.columns for c in TS_FEATURES)
        print(f"  Time-series features: {'yes' if has_ts else 'no'}")

        # 2. Target
        print("\n[2/6] Creating target column ...")
        df["result"] = df.apply(get_result, axis=1)
        print(df["result"].value_counts().to_string())

        # 3. Encode
        print("\n[3/6] Encoding categorical features ...")
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

        # 4. Split
        print("\n[4/6] Splitting data ...")
        feature_cols = [c for c in (ALL_FEATURES if has_ts else BASE_FEATURES)
                        if c in df.columns]

        if "date" in df.columns:
            df = df.sort_values("date").reset_index(drop=True)

        split_idx = int(len(df) * (1 - args.test_size))
        X = df[feature_cols]
        y = df["result"]
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

        test_df = df.iloc[split_idx:].copy()
        os.makedirs(os.path.dirname(args.test_snap) or ".", exist_ok=True)
        test_df.to_csv(args.test_snap, index=False)
        print(f"  Test snapshot saved -> {args.test_snap}")

        # 5. Train
        print(f"\n[5/6] Training LightGBM classifier (AutoML winner) ...")
        from sklearn.model_selection import train_test_split as tts

        le       = LabelEncoder()
        y_tr_enc = le.fit_transform(y_train)
        y_te_enc = le.transform(y_test)

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

        X_tr, X_val, y_tr, y_val = tts(
            X_train, y_tr_enc, test_size=0.15, random_state=args.random_state
        )
        model.fit(X_tr, y_tr)

        str_classes = list(le.classes_)
        draw_idx    = str_classes.index("Draw")

        def predict_with_threshold(proba, thresh):
            preds = []
            for row in proba:
                if row[draw_idx] >= thresh:
                    preds.append("Draw")
                else:
                    idx = max((v, i) for i, v in enumerate(row) if i != draw_idx)[1]
                    preds.append(str_classes[idx])
            return preds

        best_f1, best_thresh = 0, 0.33
        for thresh in [i / 100 for i in range(15, 45)]:
            val_preds  = predict_with_threshold(model.predict_proba(X_val), thresh)
            val_labels = le.inverse_transform(y_val)
            f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh

        print(f"  Best draw threshold: {best_thresh:.2f}  (val macro-F1={best_f1:.4f})")

        model.fit(X_train, y_tr_enc)
        model.str_classes_ = le.classes_

        preds    = predict_with_threshold(model.predict_proba(X_test), best_thresh)
        acc      = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)

        print(f"\n  Accuracy  : {acc:.4f}")
        print(f"  Macro F1  : {macro_f1:.4f}")
        print(classification_report(y_test, preds))

        if has_ts:
            importances = pd.Series(model.feature_importances_, index=feature_cols)
            print("  Top 10 features:")
            for feat, imp in importances.nlargest(10).items():
                print(f"    {feat:<35s}  {imp:.4f}")

        # MLflow logging
        mlflow.log_params({
            "model_type":       "LightGBM",
            "n_estimators":     args.n_estimators,
            "num_leaves":       31,
            "learning_rate":    0.1,
            "draw_threshold":   best_thresh,
            "n_features":       len(feature_cols),
            "train_rows":       len(X_train),
            "test_rows":        len(X_test),
            "time_based_split": True,
            "automl_winner":    "LightGBM (PyCaret, sorted by Accuracy)",
            "sort_metric":      "accuracy",
        })
        mlflow.log_metrics({
            "accuracy":     round(acc,      4),
            "macro_f1":     round(macro_f1, 4),
            "val_macro_f1": round(best_f1,  4),
        })

        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        for cls in ["Away Win", "Draw", "Home Win"]:
            safe = cls.lower().replace(" ", "_")
            if cls in report:
                mlflow.log_metrics({
                    f"{safe}_precision": round(report[cls]["precision"], 4),
                    f"{safe}_recall":    round(report[cls]["recall"],    4),
                    f"{safe}_f1":        round(report[cls]["f1-score"],  4),
                })

        mlflow.set_tags({
            "dataset":   "FIFA international results",
            "n_matches": len(df),
            "features":  "time_series + encoded",
        })

        # 6. Save
        print("\n[6/6] Saving artefacts ...")
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        joblib.dump(model,        args.output)
        joblib.dump(encoders,     args.encoders)
        joblib.dump(feature_cols, "models/feature_cols.pkl")
        joblib.dump(best_thresh,  "models/draw_threshold.pkl")
        joblib.dump(le,           "models/target_encoder.pkl")
        joblib.dump("lgbm",       "models/model_type.pkl")

        mlflow.log_artifact(args.output)
        mlflow.log_artifact("models/feature_cols.pkl")

        print(f"  Model    -> {args.output}")
        print(f"  Encoders -> {args.encoders}")
        print("\nTraining complete. Run 'mlflow ui' to view experiment.")


if __name__ == "__main__":
    main()
