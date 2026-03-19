import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import io
from model_training import train_test_split_by_year, run_model, evaluate_model  # Importing functions

# Function to load CSV files
def load_csv(file_path):
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='ISO-8859-1')
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")

# Function to load and verify the model
def load_model(model_path):
    try:
        import joblib
        return joblib.load(model_path)
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

# Function to plot feature importance
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)  # ascending so most important is at top
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(importances)), importances[indices], align="center")
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(np.array(feature_names)[indices])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Features")
    ax.set_title("Feature Importance")
    plt.tight_layout()
    return fig

def _xgb_shap_contribs(model, X, feature_names):
    """Compute SHAP values using XGBoost native pred_contribs (avoids SHAP/XGBoost version conflicts)."""
    dmat = xgb.DMatrix(X, feature_names=list(feature_names))
    contribs = model.get_booster().predict(dmat, pred_contribs=True)
    shap_vals = contribs[:, :-1]   # (n_samples, n_features)
    bias = contribs[0, -1]         # base value (same for all rows in regression)
    return shap_vals, bias

# Function to plot SHAP summary plot
def plot_shap_summary(model, X_test, feature_names):
    shap_vals, _ = _xgb_shap_contribs(model, X_test, feature_names)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_vals, X_test, feature_names=list(feature_names), show=False)
    return fig

# Function to generate SHAP force plots
def generate_shap_force_plots(model, X_test, top_indices, feature_names):
    shap_vals, bias = _xgb_shap_contribs(model, X_test, feature_names)
    html_plots = []
    for idx in top_indices:
        force_plot_html = shap.force_plot(
            bias,
            shap_vals[idx],
            X_test[idx],
            feature_names=list(feature_names),
            show=False
        )
        buf = io.StringIO()
        shap.save_html(buf, force_plot_html)
        buf.seek(0)
        html_plots.append(buf.getvalue())
    return html_plots

def get_headshot_url(player_name):
    """Return NBA CDN headshot URL for a player name, or None if not found."""
    try:
        from nba_api.stats.static import players as nba_players
        results = nba_players.find_players_by_full_name(player_name)
        if results:
            return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{results[0]['id']}.png"
    except Exception:
        pass
    return None

def find_similar_seasons(player_row, master_df, feat_cols, n=3):
    """Find the n historical MVP winner seasons most similar to player_row.

    Similarity is measured by Euclidean distance on normalized feature values.
    Only seasons where the player had the highest vote share in that year are considered.
    """
    from sklearn.preprocessing import StandardScaler

    # One winner per year: player with highest Share
    winner_idx = master_df.groupby('year')['Share'].idxmax()
    mvp_winners = master_df.loc[winner_idx].copy()

    # Restrict to columns available in both datasets
    avail_cols = [c for c in feat_cols if c in mvp_winners.columns]
    mvp_winners = mvp_winners.dropna(subset=avail_cols).reset_index(drop=True)

    # Build current player's feature vector
    current_vec = []
    for c in avail_cols:
        try:
            current_vec.append(float(player_row.get(c, 0) or 0))
        except (TypeError, ValueError):
            current_vec.append(0.0)

    hist_matrix = mvp_winners[avail_cols].values.astype(float)

    # Normalize using both historical + current together
    scaler = StandardScaler()
    all_data = np.vstack([hist_matrix, current_vec])
    all_scaled = scaler.fit_transform(all_data)
    hist_scaled = all_scaled[:-1]
    current_scaled = all_scaled[-1]

    distances = np.sqrt(((hist_scaled - current_scaled) ** 2).sum(axis=1))
    top_indices = np.argsort(distances)[:n]

    results = []
    for idx in top_indices:
        row = mvp_winners.iloc[idx]
        results.append({
            'Player': row['Player'],
            'year': int(row['year']),
            'Share': float(row['Share']),
            'similarity': max(0.0, 1.0 - distances[idx] / (distances.max() + 1e-9)),
        })
    return results


# Function to apply color formatting
def highlight_cells(val):
    color = 'background-color: lightgreen' if val == 'correct' else ('background-color: lightcoral' if val == 'incorrect' else '')
    return color

# Function to calculate accuracy percentage
def calculate_accuracy_percentage(df):
    correct_predictions = df['Label'].value_counts().get('correct', 0)
    total_predictions = len(df)
    return (correct_predictions / total_predictions) * 100

# Function to prepare data for evaluation
def prepare_evaluation_data(df, year, model):
    df_year = df[df['year'] == year]
    X_train, y_train, X_test, y_test, feature_names = train_test_split_by_year(year, df)
    predictions, _, predicted_mvp, actual_mvp = run_model(model, X_train, y_train, X_test, y_test, df, year)
    mse, r2 = evaluate_model(predictions, y_test)
    return df_year, X_test, feature_names, predictions, predicted_mvp, actual_mvp, mse, r2
