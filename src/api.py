"""
api.py
======
FastAPI server for FIFA 2026 match prediction.

Usage:
    pip install fastapi uvicorn
    python src/api.py

Endpoints:
    GET  /health          — liveness check
    POST /predict         — predict match result
    GET  /model/info      — model metadata
    GET  /monitor/latest  — latest monitoring report
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FIFA 2026 MLOps API",
    description="Match result prediction pipeline — RandomForest with time-series features",
    version="1.0.0",
)

# Allow the demo.html to call the API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve demo.html at the root
@app.get("/")
def serve_demo():
    demo_path = os.path.join(os.path.dirname(__file__), "..", "demo.html")
    if os.path.exists(demo_path):
        return FileResponse(demo_path)
    return {"message": "FIFA 2026 MLOps API — see /docs"}

# ─── Load artefacts on startup ────────────────────────────────────────────────

MODEL_PATH    = "models/fifa_model.pkl"
ENCODERS_PATH = "models/encoders.pkl"
FEAT_PATH     = "models/feature_cols.pkl"
THRESH_PATH   = "models/draw_threshold.pkl"
MONITOR_PATH  = "logs/monitoring_report.json"

model        = None
encoders     = None
feature_cols = None
draw_thresh  = 0.28
model_loaded_at = None


@app.on_event("startup")
def load_model():
    global model, encoders, feature_cols, draw_thresh, model_loaded_at
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at '{MODEL_PATH}'. Run train.py first.")
    model        = joblib.load(MODEL_PATH)
    encoders     = joblib.load(ENCODERS_PATH)
    feature_cols = joblib.load(FEAT_PATH) if os.path.exists(FEAT_PATH) else None
    draw_thresh  = joblib.load(THRESH_PATH) if os.path.exists(THRESH_PATH) else 0.28
    model_loaded_at = datetime.utcnow().isoformat()
    print(f"  ✅  Model loaded — draw threshold: {draw_thresh}")


# ─── Schemas ──────────────────────────────────────────────────────────────────

class MatchRequest(BaseModel):
    home_team:  str
    away_team:  str
    tournament: str = "FIFA World Cup"
    neutral:    bool = False


class PredictionResponse(BaseModel):
    home_team:   str
    away_team:   str
    tournament:  str
    neutral:     bool
    prediction:  str
    probability: dict
    model_version: str = "1.0.0"
    timestamp:   str


# ─── Helpers ──────────────────────────────────────────────────────────────────

def encode_team(team: str, encoder_key: str) -> int:
    enc   = encoders[encoder_key]
    known = set(enc.classes_)
    if team not in known:
        # Unknown team — return median encoded value as fallback
        return int(np.median(enc.transform(enc.classes_)))
    return int(enc.transform([team])[0])


def build_feature_row(req: MatchRequest) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame.
    Time-series features are unavailable at inference time without match history,
    so we fill them with the training set means (neutral prior).
    """
    FEATURE_MEANS = {
        "home_win_rate": 0.391, "home_draw_rate": 0.228, "home_loss_rate": 0.381,
        "home_goals_scored_avg": 1.482, "home_goals_conceded_avg": 1.445,
        "home_home_win_rate": 0.490, "home_home_goals_avg": 1.759,
        "home_streak": 0.0,
        "away_win_rate": 0.382, "away_draw_rate": 0.227, "away_loss_rate": 0.391,
        "away_goals_scored_avg": 1.455, "away_goals_conceded_avg": 1.487,
        "away_away_win_rate": 0.281, "away_away_goals_avg": 1.181,
        "away_streak": 0.0,
        "h2h_home_win_rate": 0.415,
        "win_rate_diff": 0.009, "goals_diff": 0.027, "streak_diff": 0.096,
    }

    row = {
        "home_team_encoded":  encode_team(req.home_team,  "home_team"),
        "away_team_encoded":  encode_team(req.away_team,  "away_team"),
        "tournament_encoded": encode_team(req.tournament, "tournament"),
        "neutral":            int(req.neutral),
    }

    if feature_cols:
        for col in feature_cols:
            if col not in row:
                row[col] = FEATURE_MEANS.get(col, 0.0)
        return pd.DataFrame([row])[feature_cols]

    return pd.DataFrame([row])


def predict_result(row: pd.DataFrame) -> tuple[str, dict]:
    """Run prediction with draw threshold, return label + probabilities."""
    classes = list(model.classes_)

    try:
        proba    = model.predict_proba(row)[0]
        draw_idx = classes.index("Draw")

        prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

        if proba[draw_idx] >= draw_thresh:
            label = "Draw"
        else:
            idx   = max((v, i) for i, v in enumerate(proba) if i != draw_idx)[1]
            label = classes[idx]

    except Exception:
        # PyCaret pipeline fallback
        preds = model.predict(row)
        label = str(preds[0])
        prob_dict = {"Home Win": 0.0, "Draw": 0.0, "Away Win": 0.0}
        prob_dict[label] = 1.0

    return label, prob_dict


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_loaded_at": model_loaded_at,
        "draw_threshold": draw_thresh,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: MatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if req.home_team == req.away_team:
        raise HTTPException(status_code=400, detail="Home and away teams must differ")

    row = build_feature_row(req)
    label, prob_dict = predict_result(row)

    return PredictionResponse(
        home_team=req.home_team,
        away_team=req.away_team,
        tournament=req.tournament,
        neutral=req.neutral,
        prediction=label,
        probability=prob_dict,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/model/info")
def model_info():
    info = {
        "model_type": type(model).__name__ if model else None,
        "feature_count": len(feature_cols) if feature_cols else 4,
        "features": feature_cols,
        "draw_threshold": draw_thresh,
        "loaded_at": model_loaded_at,
    }
    if hasattr(model, "n_estimators"):
        info["n_estimators"] = model.n_estimators
    return info


@app.get("/monitor/latest")
def monitor_latest():
    if not os.path.exists(MONITOR_PATH):
        raise HTTPException(status_code=404,
                            detail="No monitoring report found. Run monitor.py first.")
    with open(MONITOR_PATH) as f:
        reports = json.load(f)
    return reports[-1] if reports else {}


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
