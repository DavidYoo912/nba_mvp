# mvp_projection.py

import pandas as pd
from utils import load_model

# Exact features the model was trained on — must match TO_DROP exclusions in model_training.py.
# If TO_DROP is changed and the model is retrained, update this list to match.
FEATURE_COLUMNS = [
    'Age', 'PTS', 'W/L%', 'seed', 'player_efficiency_rating',
    'free_throw_attempt_rate', 'usage_percentage',
    'win_shares_per_48_minutes',
    'offensive_box_plus_minus',
    'value_over_replacement_player',
]

DISPLAY_STATS = ['Player', 'PTS', 'W/L%', 'win_shares_per_48_minutes', 'value_over_replacement_player',
                 'offensive_box_plus_minus', 'seed', 'predictions']


def predict_mvp(players_data_path, model_path, n=10):
    """Load current candidates and return ranked predictions.

    Returns:
        all_candidates: DataFrame sorted by predictions descending.
        feat_cols: list of feature column names used for prediction.
    """
    model = load_model(model_path)
    players_df = pd.read_csv(players_data_path)

    feat_cols = [c for c in FEATURE_COLUMNS if c in players_df.columns]
    X = players_df[feat_cols]
    players_df['predictions'] = model.predict(X)

    all_candidates = players_df.sort_values('predictions', ascending=False).reset_index(drop=True)
    return all_candidates, feat_cols
