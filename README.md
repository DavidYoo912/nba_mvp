# NBA MVP Predictor

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=python&logoColor=white)
![last commit](https://img.shields.io/github/last-commit/davidyoo912/nba_mvp?color=orange)
![forks](https://img.shields.io/github/forks/DavidYoo912/nba_mvp?style=social)

A machine learning system that predicts NBA MVP vote share using 45+ years of historical data (1980–present). Includes a live in-season tracker and a full historical back-test evaluation — both surfaced through an interactive Streamlit app.

---

## How It Works

The model is trained on every MVP candidate season from 1980 to present. It learns which combinations of player and team stats historically correlate with high MVP vote share, then applies that to the current season's candidates.

**Validation** uses leave-one-year-out cross-validation: to evaluate any given season, the model is trained on all *other* seasons — it never sees the year it's being tested on. This gives an honest estimate of real-world accuracy.

**Result: 78.3% accuracy (36/46 seasons correct) on historical back-testing.**

---

## Features

### In-Season Tracker
- Predicted MVP vote share for all current candidates
- Player headshots on the bar chart
- Race tightness indicator (Clear Frontrunner / Moderate Lead / Tight Race)
- Season trend chart — Polymarket-style lines tracking each player's predicted share over time
- Per-player deep dive: SHAP force plot, key stats, what-if sliders, historical comps

### Historical Evaluation
- Season-by-season back-test results (1980–present)
- Career MVP vote share history with crown markers on actual win years and selected-year highlight
- SHAP explanations for the top-3 predicted candidates
- Feature importance chart

---

## Model

| | |
|---|---|
| **Algorithm** | XGBoost Regressor |
| **Target** | `Share` — fraction of total MVP votes (0–1) |
| **Validation** | Leave-one-year-out cross-validation |
| **Accuracy** | 78.3% (36/46 seasons, correct MVP winner predicted) |
| **Avg MSE** | 0.0263 |
| **Avg R²** | 0.623 |

**Features used:** Points per game · Team W/L% · Conference seed · PER · Free throw attempt rate · Usage % · Win Shares per 48 min · Offensive BPM · VORP · Age

---

## Project Structure

```
nba_mvp/
├── nba_mvp_project/        # Main application code
│   ├── app.py              # Streamlit app (two tabs)
│   ├── evaluation.py       # Historical evaluation UI components
│   ├── mvp_projection.py   # In-season prediction logic
│   ├── model_training.py   # Grid search + leave-one-year-out CV
│   ├── extract_historical.py  # Scrape historical data from Basketball Reference
│   ├── extract_current.py     # Scrape current season candidates
│   ├── scraper_utils.py    # Shared scraping helpers
│   └── utils.py            # Model loading, SHAP, feature importance
├── data/
│   ├── master_table.csv           # Historical candidates (1980–present)
│   ├── current_candidate_table.csv # Current season candidates
│   ├── best_model_summary.csv     # Per-year back-test results
│   └── prediction_history.csv     # Time-series of in-season predictions
├── model/
│   └── best_model.pkl      # Trained XGBoost model (joblib)
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
pip install basketball-reference-web-scraper lightgbm plotly beautifulsoup4
```

---

## Usage

### Run the app
```bash
cd nba_mvp_project
streamlit run app.py
```

### Update data
```bash
# Historical training data (1980–present)
cd nba_mvp_project && python extract_historical.py

# Current season candidates + append today's predictions to prediction_history.csv
cd nba_mvp_project && python extract_current.py
```

### Retrain the model
```bash
cd nba_mvp_project
python model_training.py
```
Runs a hyperparameter grid search with leave-one-year-out CV and saves the best model to `model/best_model.pkl`.

---

## Data Sources

All data scraped from [Basketball Reference](https://www.basketball-reference.com/) — MVP voting history, team standings, and advanced stats. Scripts include 3-second delays between requests to respect rate limits.
