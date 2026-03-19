# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Run the Streamlit app
```bash
cd nba_mvp_project
streamlit run app.py
```

### Train / retrain the model
```bash
cd nba_mvp_project
python model_training.py
```
Runs grid search, leave-one-year-out CV, saves `model/best_model.pkl` and `data/best_model_summary.csv`.

### Update data
```bash
# Historical MVP data (1980–present) → data/master_table.csv
cd nba_mvp_project && python extract_historical.py

# Current season candidates → data/current_candidate_table.csv
cd nba_mvp_project && python extract_current.py
```
Both scripts add 3-second delays between requests to avoid rate-limiting Basketball Reference.

### Install dependencies
```bash
pip install -r requirements.txt
# Also required (not in requirements.txt):
pip install basketball-reference-web-scraper lightgbm plotly beautifulsoup4
```

## Architecture

### ML Pipeline

```
extract_historical.py  ──► data/master_table.csv ──► model_training.py ──► model/best_model.pkl
extract_current.py     ──► data/current_candidate_table.csv
                                                           │
                                         app.py (Streamlit) ◄──────────────────────────────┘
                                          ├── evaluation.py   (historical year-by-year tab)
                                          └── mvp_projection.py (in-season tracker tab)
```

**Task**: Regression on `Share` (MVP vote share, 0–1). The player with the highest predicted share in a given year is the predicted MVP winner.

**Model**: XGBoost Regressor. LightGBM/Ridge ensemble code exists but is commented out.

**Validation**: Leave-one-year-out cross-validation — train on all years except year Y, test on year Y. This is in `model_training.py:train_test_split_by_year`.

### Module Responsibilities

| Module | Role |
|---|---|
| `model_training.py` | Grid search, leave-one-year-out CV, `TO_DROP` feature list, saves model |
| `extract_historical.py` | Scrapes Basketball Reference: MVP votes, team standings, advanced stats (incremental) |
| `extract_current.py` | Scrapes current season candidates + advanced stats |
| `mvp_projection.py` | Loads model + current data, returns top-3 predictions |
| `evaluation.py` | Streamlit UI components: per-year metrics, SHAP force plots |
| `utils.py` | CSV/model loading (joblib), feature importance plots, SHAP summary plots, accuracy helpers |
| `app.py` | Two-tab Streamlit app wiring evaluation.py and mvp_projection.py |

### Key Data Files

- `data/master_table.csv` — primary training dataset (historical candidates 1980+, merged team standings + advanced stats)
- `data/current_candidate_table.csv` — current season candidates for projection
- `data/best_model_summary.csv` — per-year evaluation results (MSE, R², predicted vs. actual MVP)
- `model/best_model.pkl` — serialized best XGBoost model (joblib)
- `nba_mvp_project/grid_results.csv` — hyperparameter grid search results

### Feature Selection

Features are selected by exclusion: the `TO_DROP` list in `model_training.py` specifies which columns to drop. Columns **not** in `TO_DROP` become model features. Currently kept: Age, PTS, W/L%, seed, PER, free_throw_attempt_rate, usage_percentage, win_shares_per_48_minutes, offensive_box_plus_minus, value_over_replacement_player.

### Path Conventions

All modules in `nba_mvp_project/` are designed to be run from that directory. They reference sibling directories (`data/`, `model/`) using `os.path.dirname(os.getcwd())`. The Streamlit app (`streamlit run app.py`) must also be launched from `nba_mvp_project/`.

### Import Dependencies

`utils.py` ← `evaluation.py` ← `app.py` and `model_training.py` ← `utils.py`. Avoid introducing circular imports by keeping this order.
