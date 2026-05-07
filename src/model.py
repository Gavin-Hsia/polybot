"""Train and load the match prediction model."""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from features import load_matches, StatsCache, build_training_rows, build_live_features, FEATURE_COLS

MODEL_PATH   = Path(__file__).parent.parent / "data" / "model.pkl"
CACHE_PATH   = Path(__file__).parent.parent / "data" / "stats_cache.pkl"
TRAIN_CUTOFF = "2023-01-01"
VAL_CUTOFF   = "2024-01-01"


def train(verbose: bool = True):
    df = load_matches()

    train_df = df[df["tourney_date"] < TRAIN_CUTOFF]
    val_df   = df[(df["tourney_date"] >= TRAIN_CUTOFF) & (df["tourney_date"] < VAL_CUTOFF)]
    test_df  = df[df["tourney_date"] >= VAL_CUTOFF]

    if verbose:
        print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
        print("Building stats cache (one-time precomputation)...")

    cache = StatsCache(df)

    # Save cache so scan.py can reuse it without recomputing
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    if verbose:
        print("Building training features...")

    X_train, y_train = build_training_rows(train_df, cache)
    X_val,   y_val   = build_training_rows(val_df,   cache)
    X_test,  y_test  = build_training_rows(test_df,  cache)

    if verbose:
        print(f"Training XGBoost on {len(X_train):,} examples...")

    base = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    model.fit(X_train, y_train)

    val_acc    = (model.predict(X_val)  == y_val).mean()
    test_acc   = (model.predict(X_test) == y_test).mean()
    val_brier  = float(((model.predict_proba(X_val)[:, 1]  - y_val)  ** 2).mean())
    test_brier = float(((model.predict_proba(X_test)[:, 1] - y_test) ** 2).mean())

    if verbose:
        print(f"\nVal  accuracy: {val_acc:.1%}  Brier: {val_brier:.4f}")
        print(f"Test accuracy: {test_acc:.1%}  Brier: {test_brier:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    meta = {"val_accuracy": val_acc, "test_accuracy": test_acc,
            "val_brier": val_brier, "test_brier": test_brier}
    with open(MODEL_PATH.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"\nModel saved → {MODEL_PATH}")
        print(f"Cache saved → {CACHE_PATH}")

    return model, cache, meta


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not trained — run: python train.py")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_cache():
    if not CACHE_PATH.exists():
        raise FileNotFoundError("Cache missing — run: python train.py")
    with open(CACHE_PATH, "rb") as f:
        return pickle.load(f)


def predict_live(p1: str, p2: str, surface: str) -> float:
    """Returns P(p1 wins) for a live match."""
    df    = load_matches()
    model = load_model()
    cache = load_cache()
    X = build_live_features(p1, p2, surface, df, cache)
    return float(model.predict_proba(X)[0, 1])
