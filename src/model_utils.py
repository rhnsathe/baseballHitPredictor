import os
import joblib
import numpy as np

MODEL_PATH = "gbc_matchup_model.pkl"

def load_model_utils():
    """
    Returns:
      gbc,
      batter_stats,
      pitcher_stats,
      feature_cols,
      best_thresh,
      bat_season_map,
      pit_season_map
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found; run train_model.py first.")
    return joblib.load(MODEL_PATH)


def predict_matchup(
    b_id: int,
    p_id: int,
    gbc,
    batter_stats: dict,
    pitcher_stats: dict,
    feature_cols: list,
    bat_season_map: dict,
    pit_season_map: dict
) -> float:
    """
    Build a single‐PA feature vector for (b_id vs p_id) and return P(hit).
    """
    bs = batter_stats.get(b_id)
    ps = pitcher_stats.get(p_id)
    if bs is None or ps is None:
        raise ValueError(f"Missing stats for batter {b_id} or pitcher {p_id}")

    feat = {}

    # zone × metric averages
    for z,cell in bs["zone_metrics_agg"].items():
        for m,v in cell.items():
            feat[f"bat_z{z}_{m}"] = v
    for z,cell in ps["zone_metrics_agg"].items():
        for m,v in cell.items():
            feat[f"pit_z{z}_{m}"] = v

    # distributions & overall‐zone %
    for pt,d in bs["distribution"].items():
        feat[f"bat_pct_{pt}"] = d["pct_of_player"]
    for z,pct in bs["overall_zone_pct"].items():
        feat[f"bat_pct_z{z}"] = pct
    for pt,d in ps["distribution"].items():
        feat[f"pit_pct_{pt}"] = d["pct_of_player"]
    for z,pct in ps["overall_zone_pct"].items():
        feat[f"pit_pct_z{z}"] = pct

    # seasonal stats
    for stat,val in bat_season_map.get(b_id, {}).items():
        feat[f"bat_season_{stat}"] = val
    for stat,val in pit_season_map.get(p_id, {}).items():
        feat[f"pit_season_{stat}"] = val

    X_new = np.array([[feat.get(c, 0.0) for c in feature_cols]])
    return float(gbc.predict_proba(X_new)[0, 1])