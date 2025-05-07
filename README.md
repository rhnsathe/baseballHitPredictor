# Pre‑At‑Bat Hit Probability Model

A Python project that builds a lightweight “pre‑at‑bat” probability estimator for whether a plate appearance will result in a hit, based only on historical batter vs. pitcher statistics and season‑to‑date metrics. It uses a `GradientBoostingClassifier` under the hood to output well‑calibrated hit‑probabilities.

---

## Features

- **In‑place feature engineering**  
  – Zone-by-zone hit counts, distributions, and aggregated metrics for batters & pitchers  
  – Per‑player seasonal statistics from CSV files  
- **Balanced training**  
  – Upsamples minority (“hit”) class for a balanced training set  
- **Probability outputs**  
  – Returns `P(hit | batter, pitcher)` via `predict_proba`  
- **Threshold tuning**  
  – Finds an optimal precision–recall threshold on a held‑out validation split  
- **Easy CLI**  
  – `train_model.py` to train & serialize model + stats  
  – `evaluate_model.py` to report ROC‑AUC, precision, recall, F1, and full classification report  
- **Re‑usable utilities**  
  – `model_utils.predict_matchup(...)` for one‑off probability predictions  
  – Single pickle file (`gbc_matchup_model.pkl`) contains the trained model, lookup tables, feature list, and best threshold

---

## Requirements

- Python 3.7+
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [scikit‑learn](https://scikit-learn.org/)
- [joblib](https://joblib.readthedocs.io/)

```bash
pip install pandas numpy scikit-learn joblib
