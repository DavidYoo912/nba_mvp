import json
import os
import time
import warnings
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scraper_utils import ADVANCED_STATS_RENAME, fetch_page, extract_advanced_stats

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
data_path = ROOT / 'data'

with open(ROOT / 'scripts' / 'team_to_abbreviations.json') as f:
    team_to_abbreviations = json.load(f)


def extract_mvp_candidates(year):
    url = f"https://www.basketball-reference.com/awards/awards_{year}.html"
    try:
        html = fetch_page(url)
        mvp_candidate_table = pd.read_html(StringIO(html))[0].droplevel(level=0, axis=1)
        mvp_candidate_table['year'] = year
    except Exception as err:
        print(f'No MVP race data found for year {year}: {err}')
        return None
    return mvp_candidate_table


def extract_team_stats(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_standings.html"
    html = fetch_page(url)
    tables = pd.read_html(StringIO(html))

    team_east = tables[0].rename({'Eastern Conference': 'team'}, axis=1)
    team_east = team_east[team_east['team'].str.contains('Division') == False]
    team_east['seed'] = team_east['W'].rank(ascending=False)

    team_west = tables[1].rename({'Western Conference': 'team'}, axis=1)
    team_west = team_west[team_west['team'].str.contains('Division') == False]
    team_west['seed'] = team_west['W'].rank(ascending=False)

    standings = pd.concat([team_east, team_west])
    standings.team = standings.team.str.replace('*', '')
    standings['Tm'] = standings['team'].map(team_to_abbreviations)
    return standings[['Tm', 'team', 'W', 'W/L%', 'seed']]


def extract_historical_table():
    current_year = date.today().year
    years = list(range(1980, current_year))

    full_path_historic = data_path / 'master_table.csv'

    if full_path_historic.exists():
        print(f"Loading existing master_table.csv from {full_path_historic}")
        master_table_existing = pd.read_csv(full_path_historic)
        existing_years = master_table_existing['year'].unique()
        years_to_extract = [y for y in years if y not in existing_years]
    else:
        print("No existing master_table.csv found. Creating new file.")
        years_to_extract = years

    print(f'Extracting data for years: {years_to_extract}')
    tables = []
    for year in tqdm(years_to_extract):
        time.sleep(3)  # Avoid hitting rate limits on Basketball Reference

        mvp_candidate_table = extract_mvp_candidates(year)
        if mvp_candidate_table is None:
            continue

        team_standing_table_sub = extract_team_stats(year)
        table = pd.merge(mvp_candidate_table, team_standing_table_sub, how='left', on='Tm')

        # Historical data does not include Age (not in master_table feature set)
        adv = extract_advanced_stats(year, include_age=False)
        table = pd.merge(table, adv, how='left', on='Player')

        tables.append(table)

    if tables:
        master_table_new = pd.concat(tables)
        master_table_new = master_table_new[master_table_new['Tm'].str.contains('TOT') == False]
        master_table_new['3P%'] = master_table_new['3P%'].fillna(0)

        if full_path_historic.exists():
            master_table_existing = pd.read_csv(full_path_historic)
            master_table_combined = pd.concat([master_table_existing, master_table_new])
            master_table_combined.to_csv(full_path_historic, index=False)
            print("Data appended to existing master_table.csv")
        else:
            master_table_new.to_csv(full_path_historic, index=False)
            print("New master_table.csv created")

    print('Extraction complete.')


if __name__ == "__main__":
    extract_historical_table()
