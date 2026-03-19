# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

NBA MVP prediction model using XGBoost regression. The model predicts MVP vote share (`Share` column) for each candidate, then picks the player with the highest predicted share as the MVP winner. Historical data from 1980–present is scraped from Basketball Reference. A Streamlit app provides evaluation and in-season projection views.

## Commands

### Install dependencies
```
pip install -r requirements.txt
pip install basketball-reference-web-scraper lightgbm plotly beautifulsoup4
```
Note: `requirements.txt` does not include all dependencies. The additional packages above are required by the extraction and training modules.

### Run the Streamlit app
```
cd nba_mvp_project && streamlit run app.py
```
The app expects to be run from inside `nba_mvp_project/` because paths are resolved relative to `os.getcwd()` (e.g., `os.path.dirname(os.getcwd())` to reach `data/` and `model/`).

### Train the model
```
cd nba_mvp_project && python model_training.py
```
Runs grid search, trains the best model, saves it to `model/best_model.pkl`, and writes evaluation summary to `data/best_model_summary.csv`.

### Extract data
```
cd nba_mvp_project && python extract_historical.py   # historical MVP candidate data → data/master_table.csv
cd nba_mvp_project && python extract_current.py      # current season candidates → data/current_candidate_table.csv
```
Extraction scripts scrape Basketball Reference with `time.sleep(3)` between requests to avoid rate limiting. They must also be run from inside `nba_mvp_project/`.

## Architecture

### Data pipeline
1. **Extraction** (`extract_historical.py`, `extract_current.py`): Scrapes MVP candidate stats, team standings, and advanced stats from Basketball Reference. Merges them into a single table per year. Historical data is appended incrementally to `data/master_table.csv`.
2. **Training** (`model_training.py`): Uses leave-one-year-out cross-validation — for each year, all other years are training data. The `TO_DROP` list at the top of this file controls which columns are excluded from features; commented-out entries in that list are the key features kept for modeling. `custom_grid_search` evaluates hyperparameter combinations by accuracy (correct MVP prediction) then R².
3. **Evaluation/App** (`app.py`, `evaluation.py`, `utils.py`): Streamlit UI with two tabs — "Evaluation" (historical year-by-year analysis with SHAP explanations) and "In Season Tracker" (current season MVP projection via `mvp_projection.py`).

### Key design details
- **Target variable**: `Share` (MVP vote share percentage), not a binary classification.
- **Path convention**: All modules assume they are run from `nba_mvp_project/` and use `os.path.dirname(os.getcwd())` to reference sibling directories (`data/`, `model/`). Do not change working directory assumptions without updating all path references.
- **Feature selection**: Controlled by the `TO_DROP` list in `model_training.py`. Features that are commented out in the list are intentionally kept (e.g., `win_shares_per_48_minutes`, `W/L%`, `usage_percentage`, `PTS`).
- **Model persistence**: The trained model is saved as `model/best_model.pkl` via joblib. The Streamlit app and `mvp_projection.py` load this pickle at startup.
- **Circular import note**: `utils.py` imports from `model_training.py` and `evaluation.py` imports from `utils.py`. Keep this dependency order in mind when refactoring.

### Data files
- `data/master_table.csv`: Historical MVP candidates with merged stats (primary training data)
- `data/current_candidate_table.csv`: Current season MVP candidates for projection
- `data/best_model_summary.csv`: Per-year evaluation results from the best model
- `scripts/team_to_abbreviations.json`: Team name → abbreviation mapping used during extraction
