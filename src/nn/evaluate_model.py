import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from model_utils import load_model_utils

def evaluate():
    (model,
     batter_stats,
     pitcher_stats,
     feature_cols,
     best_thresh,
     bat_season_map,
     pit_season_map) = load_model_utils()

    df = pd.read_csv("../hello.csv")
    df = df.dropna(subset=["events"])
    df["label"] = df["events"].isin(
        ["single","double","triple","home_run"]
    ).astype(int)

    def make_feature_vector(r):
        b,p = r["batter"], r["pitcher"]
        bs,ps = batter_stats.get(b), pitcher_stats.get(p)
        if bs is None or ps is None:
            return None
        feat = {}

        for z,cell in bs["zone_metrics_agg"].items():
            for m,v in cell.items():
                feat[f"bat_z{z}_{m}"] = v
        for z,cell in ps["zone_metrics_agg"].items():
            for m,v in cell.items():
                feat[f"pit_z{z}_{m}"] = v
        for pt,d in bs["distribution"].items():
            feat[f"bat_pct_{pt}"] = d["pct_of_player"]
        for z,pct in bs["overall_zone_pct"].items():
            feat[f"bat_pct_z{z}"] = pct
        for pt,d in ps["distribution"].items():
            feat[f"pit_pct_{pt}"] = d["pct_of_player"]
        for z,pct in ps["overall_zone_pct"].items():
            feat[f"pit_pct_z{z}"] = pct
        for stat,val in bat_season_map.get(b, {}).items():
            feat[f"bat_season_{stat}"] = val
        for stat,val in pit_season_map.get(p, {}).items():
            feat[f"pit_season_{stat}"] = val
        return feat

    feats = [make_feature_vector(r) for _,r in df.iterrows()]
    feats = [f for f in feats if f is not None]
    data  = pd.DataFrame(feats).fillna(0)

    X       = data[feature_cols].values
    y       = df["label"].values

    y_proba = model.predict(X).ravel()
    y_pred  = (y_proba >= best_thresh).astype(int)

    print(f"ROC AUC   = {roc_auc_score(y, y_proba):.3f}")
    print(f"Precision = {precision_score(y, y_pred):.3f}")
    print(f"Recall    = {recall_score(y, y_pred):.3f}")
    print(f"F1 Score  = {f1_score(y, y_pred):.3f}\n")
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    evaluate()
