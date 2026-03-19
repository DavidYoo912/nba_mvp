import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tqdm import tqdm

warnings.filterwarnings('ignore', category=FutureWarning)

ROOT = Path(__file__).parent.parent

# Columns excluded from model features.
# Kept features (not in TO_DROP, not 'Share'):
#   Age, PTS, W/L%, seed, player_efficiency_rating,
#   free_throw_attempt_rate, usage_percentage,
#   win_shares_per_48_minutes, offensive_box_plus_minus,
#   value_over_replacement_player
TO_DROP = [
    'Rank', 'Player', 'year', 'Tm', 'team', 'First', 'Pts Won', 'Pts Max', 'WS',
    'MP', 'G', 'W',
    'FG%', '3P%', 'STL', 'BLK',
    'three_point_attempt_rate', 'total_rebound_percentage',
    'offensive_rebound_percentage', 'block_percentage',
    'defensive_rebound_percentage', 'steal_percentage',
    'turnover_percentage', 'assist_percentage',
    'AST', 'TRB', 'FT%',
    'win_shares', 'box_plus_minus',
    'defensive_box_plus_minus', 'offensive_win_shares', 'defensive_win_shares',
    'true_shooting_percentage',
    'WS/48',  # raw column; renamed version win_shares_per_48_minutes is kept
]

def train_test_split_by_year(year, df, scaling=False):
    train_df = df[df['year'] != year].drop(TO_DROP, axis=1)
    test_df = df[df['year'] == year].drop(TO_DROP, axis=1)

    if scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df.drop('Share', axis=1))
        X_test = scaler.transform(test_df.drop('Share', axis=1))
        train_df = pd.DataFrame(X_train, columns=train_df.columns.difference(['Share']))
        test_df = pd.DataFrame(X_test, columns=test_df.columns.difference(['Share']))
    else:
        X_train, X_test = train_df.drop('Share', axis=1).values, test_df.drop('Share', axis=1).values

    return (X_train, train_df['Share'].values, X_test, test_df['Share'].values, 
            train_df.columns.difference(['Share']))

def train_model(regressor, X_train, y_train):
    model = regressor.fit(X_train, y_train)
    return model

def predict_model(model, X_test, df, year):
    predictions = model.predict(X_test)
    predicted_mvp = actual_mvp = None

    if df is not None and year is not None:
        df_year = df[df['year'] == year]
        predicted_mvp = df_year.iloc[np.argmax(predictions)]['Player']
        actual_mvp = df_year.iloc[np.argmax(df_year['Share'])]['Player']

    return predictions, model, predicted_mvp, actual_mvp

def run_model(regressor, X_train, y_train, X_test, y_test, df=None, year=None):
    model = train_model(regressor, X_train, y_train)
    predictions, model, predicted_mvp, actual_mvp = predict_model(model, X_test, df, year)
    return predictions, model, predicted_mvp, actual_mvp

def evaluate_model(predictions, y_test):
    return mean_squared_error(y_test, predictions), r2_score(y_test, predictions)

def run_model_average(df, regressor, years, scaling=False, print_metrics=False, progress_bar=True):
    mse_lst = []
    r2_lst = []
    predicted_lst = []
    actual_lst = []
    label_lst = []
    model_lst = []
    cols = []

    progress = tqdm(years) if progress_bar else years

    for year in progress:
        X_train, y_train, X_test, y_test, features = train_test_split_by_year(year=year, df=df, scaling=scaling)
        predictions, model, predicted_winner, actual_winner = run_model(regressor, X_train, y_train, X_test, y_test, df=df, year=year)
        label = 'correct' if predicted_winner == actual_winner else 'incorrect'

        mse, r2 = evaluate_model(predictions, y_test)
        mse_lst.append(mse)
        r2_lst.append(r2)
        predicted_lst.append(predicted_winner)
        actual_lst.append(actual_winner)
        label_lst.append(label)
        model_lst.append(model)
        cols = features

    summary = pd.DataFrame({
        'year': years,
        'MSE': mse_lst,
        'R squared': r2_lst,
        'Predicted MVP': predicted_lst,
        'Actual MVP': actual_lst,
        'Label': label_lst
    })

    correct_count = summary['Label'].value_counts().get('correct', 0)
    incorrect_count = summary['Label'].value_counts().get('incorrect', 0)
    accuracy = correct_count / (correct_count + incorrect_count) if (correct_count + incorrect_count) > 0 else 0
    avg_mse = summary['MSE'].mean()
    avg_r2 = summary['R squared'].mean()

    if print_metrics:
        print(f"Average MSE: {avg_mse}")
        print(f"Average R squared: {avg_r2}")
        print(f"Prediction accuracy: {accuracy}")

    return avg_mse, avg_r2, accuracy, summary, model_lst, cols


def custom_grid_search(df, model_dict, param_grid_dict, years):
    results = []

    for model_name, model in model_dict.items():
        param_grid = param_grid_dict[model_name]
        param_grid_list = list(ParameterGrid(param_grid))

        with tqdm(total=len(param_grid_list), desc=f'Grid Search for {model_name}', leave=True) as pbar:
            for params in param_grid_list:
                model.set_params(**params)
                avg_mse, avg_r2, accuracy, _, _, _ = run_model_average(df, model, years, progress_bar=False)
                results.append({**params, 'model': model_name, 'MSE': avg_mse, 'R2': avg_r2, 'Accuracy': accuracy})
                pbar.update(1)

    return pd.DataFrame(results).sort_values(by=['Accuracy', 'R2'], ascending=[False, False])


def train_save_best_model(result_df, model_dict, df, years, scaling=False):
    best_result = result_df.sort_values(by=['Accuracy', 'R2'], ascending=[False, False]).iloc[0]
    best_model_name = best_result['model']
    best_params = best_result.drop(['model', 'MSE', 'R2', 'Accuracy']).to_dict()

    best_model = model_dict[best_model_name].set_params(**best_params)

    model_dir = ROOT / 'model'
    full_path_model_summary = ROOT / 'data' / 'best_model_summary.csv'
    model_save_path = model_dir / 'best_model.pkl'

    model_dir.mkdir(exist_ok=True)

    if model_save_path.exists():
        try:
            model_save_path.unlink()
            print(f"Removed existing model file: {model_save_path}")
        except Exception as e:
            print(f"Error removing existing file: {e}")

    summary_list = []
    for year in years:
        X_train, y_train, X_test, y_test, _ = train_test_split_by_year(year=year, df=df, scaling=scaling)

        best_model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)

        df_year = df[df['year'] == year]
        predicted_mvp = df_year.iloc[np.argmax(predictions)]['Player']
        actual_mvp = df_year.iloc[np.argmax(df_year['Share'])]['Player']

        label = 'correct' if predicted_mvp == actual_mvp else 'incorrect'
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        summary_list.append({
            'year': year,
            'MSE': mse,
            'R squared': r2,
            'Predicted MVP': predicted_mvp,
            'Actual MVP': actual_mvp,
            'Label': label
        })

    df_best_model_summary = pd.DataFrame(summary_list)

    try:
        joblib.dump(best_model, model_save_path)
        print(f"Best model trained and saved in '{model_save_path}'")
    except Exception as e:
        print(f"Error saving model: {e}")

    df_best_model_summary.to_csv(full_path_model_summary, index=False)
    print(f"Best model summary saved in '{full_path_model_summary}'")

    return best_model, df_best_model_summary


def main():

    stacking_model = StackingRegressor(
    estimators=[
        ('XGBoost', XGBRegressor()),
        ('LightGBM', LGBMRegressor(verbose=-1))
    ],
    final_estimator=Ridge()
)

    model_dict = {
        'XGBoost': XGBRegressor(),
        #'LightGBM': LGBMRegressor(),
        #'Stacking': stacking_model,
    }
    param_grid_dict = {
    'XGBoost': {
        'n_estimators':     [100, 200, 300],
        'max_depth':        [3, 5, 6],
        'learning_rate':    [0.05, 0.1, 0.2],
        'subsample':        [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'min_child_weight': [1, 3],
        'reg_alpha':        [0.0, 0.5],
        'reg_lambda':       [1.0, 2.0],
    },
}


    df = pd.read_csv(ROOT / 'data' / 'master_table.csv')
    years = df['year'].unique()
    results_df = custom_grid_search(df, model_dict, param_grid_dict, years)
    best_model, _ = train_save_best_model(results_df, model_dict, df, years)

    results_df.to_csv('grid_results.csv', index=False)
    print(results_df.head())

if __name__ == "__main__":
    main()
