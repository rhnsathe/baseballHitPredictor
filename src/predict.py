from src.model_utils import load_model_utils, predict_matchup

gbc, batter_stats, pitcher_stats, feature_cols, best_thresh = load_model_utils()

batter_id  = 673357
pitcher_id = 676962
prob = predict_matchup(batter_id, pitcher_id, gbc, batter_stats, pitcher_stats, feature_cols)

print(f"P(hit) = {prob:.5%}  (threshold = {best_thresh:.3f})")
