import pandas as pd
import numpy as np
import joblib
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.utils import resample

from src.model_utils import MODEL_PATH

def train_and_save():
    try:
        _ = joblib.load(MODEL_PATH)
        print("Model already exists – skipping training.")
        return
    except FileNotFoundError:
        pass

    df = pd.read_csv("hello.csv")
    df = df.dropna(subset=["events"])
    df["label"] = df["events"].isin(
        ["single","double","triple","home_run"]
    ).astype(int)

    bat_season_df = pd.read_csv("seasonbatterstats.csv")
    pit_season_df = pd.read_csv("seasonpitcherstats.csv")

    num_b = [c for c in bat_season_df.select_dtypes(include=np.number).columns if c != "player_id"]
    bat_season_df = bat_season_df[["player_id"] + num_b]
    num_p = [c for c in pit_season_df.select_dtypes(include=np.number).columns if c != "player_id"]
    pit_season_df = pit_season_df[["player_id"] + num_p]

    bat_season_map = bat_season_df.set_index("player_id").to_dict(orient="index")
    pit_season_map = pit_season_df.set_index("player_id").to_dict(orient="index")

    metrics = [
        "estimated_ba_using_speedangle",
        "estimated_woba_using_speedangle",
        "woba_value",
        "woba_denom",
        "babip_value",
        "iso_value"
    ]
    bzc = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    pzc = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    bzm = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: {"sum":{m:0.0 for m in metrics},"count":{m:0 for m in metrics}}
    )))
    pzm = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: {"sum":{m:0.0 for m in metrics},"count":{m:0 for m in metrics}}
    )))

    for _, r in df.iterrows():
        b,p = r["batter"], r["pitcher"]
        pt,z = r["pitch_type"], r["zone"]
        bzc[b][pt][z] += 1
        pzc[p][pt][z] += 1
        for m in metrics:
            v = r.get(m)
            if pd.notna(v):
                bzm[b][pt][z]["sum"][m]   += v
                bzm[b][pt][z]["count"][m] += 1
                pzm[p][pt][z]["sum"][m]   += v
                pzm[p][pt][z]["count"][m] += 1

    def to_regular(d):
        if isinstance(d, defaultdict):
            return {k: to_regular(v) for k,v in d.items()}
        return d

    def compute_averages(cell_map):
        return {
            pt:{
                z:{
                    m:(cell["sum"][m]/cell["count"][m]) if cell["count"][m]>0 else 0.0
                    for m in metrics
                }
                for z,cell in zones.items()
            }
            for pt,zones in cell_map.items()
        }

    def compute_distribution(count_map):
        total = sum(sum(z.values()) for z in count_map.values())
        dist = {}
        for pt,zones in count_map.items():
            pt_tot = sum(zones.values())
            pct_all = (pt_tot/total) if total>0 else 0.0
            zone_pct = {z:(c/pt_tot) if pt_tot>0 else 0.0 for z,c in zones.items()}
            dist[pt] = {"pct_of_player":pct_all, "zone_pct":zone_pct}
        return dist

    def compute_overall_zone_pct(count_map):
        totals = defaultdict(int)
        for zones in count_map.values():
            for z,c in zones.items():
                totals[z] += c
        s = sum(totals.values())
        return {z:(c/s if s>0 else 0.0) for z,c in totals.items()}

    def compute_zone_metrics_agg(cell_map):
        raw = defaultdict(lambda: {"sum":{m:0.0 for m in metrics},"count":{m:0 for m in metrics}})
        for zones in cell_map.values():
            for z,cell in zones.items():
                for m in metrics:
                    raw[z]["sum"][m]   += cell["sum"][m]
                    raw[z]["count"][m] += cell["count"][m]
        return {
            z:{
                m:(raw[z]["sum"][m]/raw[z]["count"][m]) if raw[z]["count"][m]>0 else 0.0
                for m in metrics
            } for z in raw
        }

    batter_stats = {
        b:{
            "zone_counts":      to_regular(zc),
            "metrics_avg":      compute_averages(bzm[b]),
            "distribution":     compute_distribution(zc),
            "overall_zone_pct": compute_overall_zone_pct(zc),
            "zone_metrics_agg": compute_zone_metrics_agg(bzm[b])
        } for b,zc in bzc.items()
    }
    pitcher_stats = {
        p:{
            "zone_counts":      to_regular(zc),
            "metrics_avg":      compute_averages(pzm[p]),
            "distribution":     compute_distribution(zc),
            "overall_zone_pct": compute_overall_zone_pct(zc),
            "zone_metrics_agg": compute_zone_metrics_agg(pzm[p])
        } for p,zc in pzc.items()
    }

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

        # seasonal stats
        for stat,val in bat_season_map.get(b, {}).items():
            feat[f"bat_season_{stat}"] = val
        for stat,val in pit_season_map.get(p, {}).items():
            feat[f"pit_season_{stat}"] = val

        return feat

    feats = [make_feature_vector(r) for _,r in df.iterrows()]
    feats = [f for f in feats if f is not None]
    data  = pd.DataFrame(feats).fillna(0)

    feature_cols = list(data.columns)
    X = data[feature_cols].values
    y = df["label"].values

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    df_tr = pd.DataFrame(X_tr, columns=feature_cols)
    df_tr["label"] = y_tr
    df_maj = df_tr[df_tr.label==0]
    df_min = df_tr[df_tr.label==1]
    df_min_up = resample(df_min, replace=True,
                         n_samples=len(df_maj), random_state=42)
    bm = pd.concat([df_maj, df_min_up])
    X_tr, y_tr = bm[feature_cols].values, bm["label"].values

    gbc = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
    )
    gbc.fit(X_tr, y_tr)

    y_va_proba = gbc.predict_proba(X_va)[:,1]
    prec, rec, th = precision_recall_curve(y_va, y_va_proba)
    best_thresh = th[np.argmax(2*prec*rec/(prec+rec+1e-10))]
    best_thresh = .50

    # 9) Save everything
    joblib.dump(
        (gbc,
         batter_stats,
         pitcher_stats,
         feature_cols,
         best_thresh,
         bat_season_map,
         pit_season_map),
        MODEL_PATH
    )
    print("Saved GB model and stats to", MODEL_PATH)

if __name__ == "__main__":
    train_and_save()