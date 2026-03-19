import json
import re
import warnings
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd

from scraper_utils import fetch_page, extract_advanced_stats

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
data_path = ROOT / 'data'

with open(ROOT / 'scripts' / 'team_to_abbreviations.json') as f:
    team_to_abbreviations = json.load(f)
abbr_to_team = {v: k for k, v in team_to_abbreviations.items()}


def extract_current_candidates():
    url = "https://www.basketball-reference.com/friv/mvp.html"
    try:
        html = fetch_page(url)
        return pd.read_html(StringIO(html))[0]
    except Exception as err:
        print(f'No MVP race data found: {err}')
        return None


def get_conference_seeds(year):
    """Return {team_abbr: conference_seed} for all 30 teams.

    Basketball Reference embeds the seed in the team name as 'Team Name (N)' when
    the playoffs picture is set; otherwise falls back to win-count ranking.
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_standings.html"
    html = fetch_page(url)
    tables = pd.read_html(StringIO(html))

    seeds = {}
    for conf_table, conf_col in [(tables[0], 'Eastern Conference'), (tables[1], 'Western Conference')]:
        df = conf_table.rename({conf_col: 'team'}, axis=1)
        df = df[~df['team'].str.contains('Division', na=False)].copy()

        # Extract embedded seed number e.g. "Boston Celtics (2)" → seed=2
        df['seed'] = df['team'].str.extract(r'\((\d+)\)').astype(float)

        df['team_clean'] = (df['team']
                            .str.replace(r'\s*\(\d+\)', '', regex=True)
                            .str.replace('*', '', regex=False)
                            .str.strip())

        df['W'] = pd.to_numeric(df['W'], errors='coerce')
        if df['seed'].isna().all():
            df['seed'] = df['W'].rank(ascending=False, method='min')

        for _, row in df.iterrows():
            abbr = team_to_abbreviations.get(row['team_clean'])
            if abbr and pd.notna(row['seed']):
                seeds[abbr] = row['seed']
    return seeds


if __name__ == "__main__":
    year = date.today().year
    full_path_current = data_path / 'current_candidate_table.csv'

    # Current-season data needs Age (it's a model feature)
    advanced_stats_df = extract_advanced_stats(year, include_age=True)

    mvp_candidate_table = extract_current_candidates()
    mvp_candidate_table = pd.merge(mvp_candidate_table, advanced_stats_df, how='left', on='Player')

    mvp_candidate_table['W'] = pd.to_numeric(mvp_candidate_table['W'], errors='coerce')
    conference_seeds = get_conference_seeds(year)
    mvp_candidate_table['seed'] = mvp_candidate_table['Team'].map(conference_seeds)

    mvp_candidate_table.to_csv(full_path_current, index=False)
    print(f"Current MVP candidate table saved: {full_path_current}")
    print(mvp_candidate_table[['Player', 'Team', 'W', 'W/L%', 'seed']])

    # Append predictions to prediction_history.csv
    model_pkl = ROOT / 'model' / 'best_model.pkl'
    if model_pkl.exists():
        import joblib
        from mvp_projection import FEATURE_COLUMNS
        model = joblib.load(model_pkl)
        feat_cols = [c for c in FEATURE_COLUMNS if c in mvp_candidate_table.columns]
        if feat_cols:
            mvp_candidate_table['predicted_share'] = model.predict(mvp_candidate_table[feat_cols])

            history_path = data_path / 'prediction_history.csv'
            today = date.today().isoformat()

            new_rows = mvp_candidate_table[['Player', 'Team', 'predicted_share']].copy()
            new_rows.insert(0, 'date', today)

            if history_path.exists():
                existing = pd.read_csv(history_path)
                existing = existing[existing['date'] != today]  # Replace if re-run today
                history = pd.concat([existing, new_rows], ignore_index=True)
            else:
                history = new_rows

            history.to_csv(history_path, index=False)
            print(f"Prediction history updated: {len(history)} total rows")
