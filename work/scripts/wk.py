"""Rajkumar's shared helpers for the work/ notebooks. Seed fixed at 42."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "outputs"; OUT.mkdir(parents=True, exist_ok=True)
EXCLUDED = {"content_id": "pseudonymous ID - grouping only", "client_id": "pseudonymous ID - grouping only",
            "trend_direction": "label source", "trend_pct": "label-derived",
            "impressions_last_30d": "trend input window", "clicks_last_30d": "trend input window",
            "sessions_last_30d": "trend input window", "impressions_prev_30d": "trend input window",
            "clicks_prev_30d": "trend input window", "sessions_prev_30d": "trend input window",
            "provider_used": "generation metadata, mostly blank", "model_used": "generation metadata, not a page property"}
NUM = ["search_volume", "competition", "cpc", "word_count", "char_count", "impressions_90d", "clicks_90d",
       "sessions_90d", "ai_sessions_90d", "days_with_impressions", "days_with_sessions", "content_age_days",
       "days_since_last_update", "ctr", "avg_position", "engagement_rate", "scroll_rate", "ai_traffic_pct"]
CAT = ["competition_level", "content_type", "main_intent"]

def load():
    df = pd.read_csv(ROOT / "data/raw/content_refresh_anonymized.csv")
    df["y"] = (df["trend_direction"] == "down").astype(int)
    return df

def features(df, num=NUM, cat=CAT):
    X = df[num].copy()
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in ["impressions_90d", "clicks_90d", "sessions_90d", "ai_sessions_90d", "search_volume"]:
        if c in X: X[c] = np.log1p(X[c].clip(lower=0))
    X = X.fillna(X.median())
    X = pd.concat([X, pd.get_dummies(df[cat].fillna("unknown"), dtype=int)], axis=1)
    return X

def client_split(df, test_size=0.2):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    tr, te = next(gss.split(df, groups=df["client_id"]))
    return tr, te

def random_split(df):
    return train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=df["y"])

def p_at_k(y, s, k):
    return float(pd.Series(np.asarray(y))[np.argsort(-np.asarray(s))[:k]].mean())

def evaluate(y, s):
    return {"roc_auc": round(roc_auc_score(y, s), 3), "avg_precision": round(average_precision_score(y, s), 3),
            "p@50": round(p_at_k(y, s, 50), 3), "p@200": round(p_at_k(y, s, 200), 3),
            "base_rate": round(float(np.mean(y)), 3), "n": int(len(y))}

def baseline_score(df):
    """My transparent rule: older, stale, visible pages that sit just off the top get reviewed first."""
    pos = pd.to_numeric(df["avg_position"], errors="coerce").fillna(100)
    s = (0.35 * (df["content_age_days"].rank(pct=True))
         + 0.25 * (df["days_since_last_update"].rank(pct=True))
         + 0.25 * np.log1p(df["impressions_90d"]).rank(pct=True)
         + 0.15 * pos.between(4, 20).astype(float))
    reasons = np.select([(df["content_age_days"] > 365) & pos.between(4, 20), df["content_age_days"] > 365,
                         pos.between(4, 20), df["days_since_last_update"] > 90],
                        ["old_page_striking_distance", "old_page", "striking_distance", "stale_update"], "monitor")
    return s, reasons

def models():
    return {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=SEED),
            "hist_gbm": HistGradientBoostingClassifier(random_state=SEED)}
