"""Shared scraping utilities for extract_historical.py and extract_current.py."""

import urllib.request
import pandas as pd
from io import StringIO


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Mapping from Basketball Reference column names to internal names
ADVANCED_STATS_RENAME = {
    'PER':   'player_efficiency_rating',
    'TS%':   'true_shooting_percentage',
    '3PAr':  'three_point_attempt_rate',
    'FTr':   'free_throw_attempt_rate',
    'ORB%':  'offensive_rebound_percentage',
    'DRB%':  'defensive_rebound_percentage',
    'TRB%':  'total_rebound_percentage',
    'AST%':  'assist_percentage',
    'STL%':  'steal_percentage',
    'BLK%':  'block_percentage',
    'TOV%':  'turnover_percentage',
    'USG%':  'usage_percentage',
    'OWS':   'offensive_win_shares',
    'DWS':   'defensive_win_shares',
    'WS':    'win_shares',
    'WS/48': 'win_shares_per_48_minutes',
    'OBPM':  'offensive_box_plus_minus',
    'DBPM':  'defensive_box_plus_minus',
    'BPM':   'box_plus_minus',
    'VORP':  'value_over_replacement_player',
}


def fetch_page(url):
    """Fetch a page with browser-like headers to avoid 403 errors."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode('utf-8')


def extract_advanced_stats(year, include_age=False):
    """Fetch advanced stats for a given season from Basketball Reference.

    Args:
        year: NBA season year (e.g. 2024 = 2023-24 season).
        include_age: If True, include the Age column (needed for current-season data).
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    html = fetch_page(url)
    advanced_stats_df = pd.read_html(StringIO(html))[0]

    # Remove repeated header rows and empty columns
    advanced_stats_df = advanced_stats_df[advanced_stats_df['Player'] != 'Player']
    advanced_stats_df = advanced_stats_df.drop(
        columns=[c for c in advanced_stats_df.columns if 'Unnamed' in str(c)],
        errors='ignore',
    )

    # Rename columns to internal names and select relevant ones
    advanced_stats_df = advanced_stats_df.rename(columns=ADVANCED_STATS_RENAME)
    base_cols = ['Player'] + (['Age'] if include_age else [])
    keep_cols = base_cols + list(ADVANCED_STATS_RENAME.values())
    advanced_stats_df = advanced_stats_df[[c for c in keep_cols if c in advanced_stats_df.columns]]

    # Convert numeric columns
    for col in advanced_stats_df.columns:
        if col != 'Player':
            advanced_stats_df[col] = pd.to_numeric(advanced_stats_df[col], errors='coerce')

    # Drop duplicate players (keep first = totals row for traded players)
    advanced_stats_df = advanced_stats_df.drop_duplicates(subset='Player', keep='first')

    return advanced_stats_df
