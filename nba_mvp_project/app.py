import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_training import train_test_split_by_year, run_model
from utils import (load_csv, load_model, generate_shap_force_plots,
                   calculate_accuracy_percentage, prepare_evaluation_data,
                   get_headshot_url, find_similar_seasons)
from evaluation import display_best_model_summary, evaluate_model_for_year, display_vote_share_trend
from mvp_projection import predict_mvp

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA MVP Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1150px; }

[data-testid="metric-container"] {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(100, 116, 139, 0.25);
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="metric-container"] label {
    font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.07em; color: #94a3b8;
}
[data-testid="stMetricValue"] { font-size: 1.65rem !important; font-weight: 700; }
[data-testid="stMetricDelta"] svg { display: none; }

[data-testid="stTabs"] [role="tablist"] {
    gap: 4px; border-bottom: 1px solid rgba(100,116,139,0.2); padding-bottom: 0;
}
[data-testid="stTabs"] [role="tab"] {
    font-size: 0.9rem; font-weight: 500;
    padding: 8px 20px; border-radius: 6px 6px 0 0; color: #94a3b8;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #f1f5f9; background: rgba(30,41,59,0.5);
    border-bottom: 2px solid #3b82f6;
}
[data-testid="stDataFrame"] th {
    font-size: 0.78rem; text-transform: uppercase;
    letter-spacing: 0.05em; color: #94a3b8;
}
h5 { color: #cbd5e1; font-weight: 600; margin: 0.5rem 0 0.25rem; }
.stCaption { color: #64748b; }

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid rgba(100,116,139,0.2) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ───────────────────────────────────────────────────────────────────────
# .resolve() guarantees an absolute path even when __file__ is relative (Streamlit quirk)
ROOT = Path(__file__).resolve().parent.parent   # nba_mvp_project/ → project root
data_path         = str(ROOT / 'data' / 'master_table.csv')
summary_csv_path  = str(ROOT / 'data' / 'best_model_summary.csv')
players_data_path = str(ROOT / 'data' / 'current_candidate_table.csv')
model_path        = str(ROOT / 'model' / 'best_model.pkl')

# ── Cached data & model loaders ────────────────────────────────────────────────
@st.cache_resource
def load_best_model(path):
    return load_model(path)

@st.cache_data
def load_historical_data(path):
    return load_csv(path)

@st.cache_data
def load_summary_data(path):
    return load_csv(path)

@st.cache_data
def load_candidates(players_path, model_path_str, file_mtime):
    """file_mtime is a real cache key — changes whenever the CSV is updated."""
    all_candidates, feat_cols = predict_mvp(players_path, model_path_str, n=10)
    return all_candidates, feat_cols

@st.cache_data
def cached_headshot(player_name):
    return get_headshot_url(player_name)


df                    = load_historical_data(data_path)
df_best_model_summary = load_summary_data(summary_csv_path)
best_model            = load_best_model(model_path)


# ── App header ─────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:1.9rem;font-weight:700;margin-bottom:0'>🏀 NBA MVP Predictor</h1>"
    "<p style='color:#64748b;margin-top:2px;margin-bottom:0.4rem'>"
    "2025-26 Season · XGBoost model trained on 45 years of NBA history (1980–2025)</p>",
    unsafe_allow_html=True,
)
with st.expander("ℹ️ How does this work?"):
    st.markdown("""
This tool uses a machine learning model to predict NBA MVP voting based on player and team statistics.

**How it was built:**
- The model was trained on historical MVP voting data from **1980 to 2025** — covering 46 seasons and hundreds of candidates.
- It learns which combinations of stats (scoring, efficiency, team success, etc.) historically lead to high MVP vote shares.
- Validation used a *leave-one-year-out* approach: to test any given season, the model was trained on all *other* seasons — meaning it never "saw" the year it was being tested on.

**What it predicts:**
- The model outputs a **predicted vote share** for each player — a number between 0 and 1 representing the estimated fraction of total MVP votes they would receive.
- The player with the highest predicted share is the model's MVP pick.

**Key stats the model uses:**
Points per game · Team win % · Win Shares per 48 min · Usage rate · PER · OBPM · VORP · Free throw attempt rate · Conference seed
""")

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_eval, tab_tracker = st.tabs(["📊  Historical Evaluation", "🏀  In-Season Tracker"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown(
        "<p style='color:#94a3b8;font-size:0.9rem;margin-bottom:1rem'>"
        "Browse 46 seasons of back-tested results. The model was tested on each season "
        "without ever having trained on it — giving an honest picture of real-world accuracy. "
        "<b>Click any row</b> to see a full breakdown of that season.</p>",
        unsafe_allow_html=True,
    )

    selected_year = None
    if df_best_model_summary is not None:
        selected_year = display_best_model_summary(df_best_model_summary)

    if selected_year is not None:
        st.markdown(f"<h4 style='margin-top:1.5rem'>Season Detail — {selected_year}</h4>",
                    unsafe_allow_html=True)
        if df is not None and best_model is not None:
            evaluate_model_for_year(df, selected_year, best_model, df_best_model_summary)
    else:
        st.markdown(
            "<p style='color:#475569;font-size:0.85rem;margin-top:4px'>"
            "↑ Click any row in the table above to explore that season in detail.</p>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — IN-SEASON TRACKER
# ══════════════════════════════════════════════════════════════════════════════
with tab_tracker:
    # Cache-busting: reload data whenever the CSV file is updated
    try:
        csv_mtime = os.path.getmtime(players_data_path)
    except OSError:
        csv_mtime = 0

    all_candidates, feat_cols = load_candidates(str(players_data_path), str(model_path), csv_mtime)

    st.markdown(
        "<p style='color:#94a3b8;font-size:0.9rem;margin-bottom:0.5rem'>"
        "Live MVP race powered by current season stats. The model scores each candidate "
        "the same way it evaluated historical seasons — the player with the highest predicted "
        "vote share is the current frontrunner. <b>Click a row</b> to see the full breakdown.</p>",
        unsafe_allow_html=True,
    )

    # Top-N slider
    col_slider, _ = st.columns([3, 7])
    with col_slider:
        top_n = st.slider("Number of candidates to show", min_value=3, max_value=10, value=5, step=1)

    display_df = all_candidates.head(top_n)

    # ── Bar chart ──────────────────────────────────────────────────────────────
    blue_shades = ['#1d4ed8', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd',
                   '#bfdbfe', '#dbeafe', '#eff6ff', '#f0f9ff', '#fafafa']
    bar_colors = blue_shades[:len(display_df)]

    # Fixed row height: each bar row is always 64px tall regardless of top_n
    row_px = 64
    chart_height = 90 + top_n * row_px
    plot_height_px = chart_height - 10 - 50   # minus top & bottom margins

    # Target headshot size: 52px square, expressed in paper units
    img_px = 52
    img_sizey = img_px / plot_height_px        # fixed pixel height → paper units
    img_sizex = 0.09                            # fixed paper-unit width (~constant visual size)
    img_x   = -0.01                             # right edge of image (just inside left margin)
    name_x  = img_x - img_sizex - 0.02         # right edge of name, left of image

    fig = go.Figure(go.Bar(
        x=display_df['predictions'].round(3),
        y=display_df['Player'],
        orientation='h',
        marker=dict(color=bar_colors),
        text=display_df['predictions'].apply(lambda x: f"{x:.3f}"),
        textposition='outside',
        cliponaxis=False,
    ))
    fig.update_layout(
        xaxis=dict(
            title="Predicted Vote Share (0 = no votes · 1 = unanimous)",
            range=[0, display_df['predictions'].max() * 1.25],
            showgrid=True, gridcolor='rgba(100,116,139,0.15)',
        ),
        # Hide default y-axis labels — replaced by annotations + images below
        yaxis=dict(autorange='reversed', showticklabels=False),
        height=chart_height,
        margin=dict(l=230, r=100, t=10, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
    )

    for i, (_, row) in enumerate(display_df.iterrows()):
        paper_y = 1.0 - (2 * i + 1) / (2 * top_n)

        headshot = cached_headshot(row['Player'])
        if headshot:
            fig.add_layout_image(
                source=headshot,
                xref='paper', yref='paper',
                x=img_x, y=paper_y,
                xanchor='right', yanchor='middle',
                sizex=img_sizex,
                sizey=img_sizey,
                layer='above',
            )

        fig.add_annotation(
            xref='paper', yref='paper',
            x=name_x, y=paper_y,
            text=row['Player'],
            xanchor='right', yanchor='middle',
            showarrow=False,
            font=dict(size=11, color='#e2e8f0'),
        )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Historical MVP winners typically receive a vote share between 0.30 and 0.95. "
        "A share near 1.0 indicates a near-unanimous MVP pick."
    )

    # ── A2: Confidence Indicator ───────────────────────────────────────────────
    if len(display_df) >= 2:
        gap = float(display_df.iloc[0]['predictions']) - float(display_df.iloc[1]['predictions'])
        leader = display_df.iloc[0]['Player']
        if gap > 0.30:
            race_label, badge_color = "CLEAR FRONTRUNNER", "#22c55e"
        elif gap > 0.10:
            race_label, badge_color = "MODERATE LEAD", "#f59e0b"
        else:
            race_label, badge_color = "TIGHT RACE", "#ef4444"
        st.markdown(
            f"<div style='margin-bottom:0.75rem'>"
            f"<span style='font-size:0.78rem;text-transform:uppercase;letter-spacing:0.07em;"
            f"color:#94a3b8'>Race Tightness</span>&nbsp;&nbsp;"
            f"<span style='background:{badge_color};color:#fff;padding:3px 12px;"
            f"border-radius:20px;font-size:0.8rem;font-weight:600'>{race_label}</span>"
            f"&nbsp;<span style='color:#64748b;font-size:0.8rem'>"
            f"Gap: {gap:.3f} ({leader} vs #{2})</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Prediction trend chart ─────────────────────────────────────────────────
    history_path = str(ROOT / 'data' / 'prediction_history.csv')
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        history_df['date'] = pd.to_datetime(history_df['date'])

        n_dates = history_df['date'].nunique()
        top_players = display_df['Player'].tolist()
        trend_df = history_df[history_df['Player'].isin(top_players)]

        if n_dates >= 2:
            fig_trend = go.Figure()
            colors = ['#1d4ed8', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd',
                      '#bfdbfe', '#dbeafe', '#eff6ff', '#f0f9ff', '#fafafa']
            for i, player in enumerate(top_players):
                pdata = trend_df[trend_df['Player'] == player].sort_values('date')
                if pdata.empty:
                    continue
                fig_trend.add_trace(go.Scatter(
                    x=pdata['date'],
                    y=pdata['predicted_share'].round(3),
                    mode='lines+markers',
                    name=player,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6),
                    hovertemplate=f'{player}<br>%{{x|%b %d}}: %{{y:.3f}}<extra></extra>',
                ))
            fig_trend.update_layout(
                title=dict(text="Predicted Vote Share — Season Trend", font=dict(size=14)),
                xaxis=dict(title="Date", tickformat="%b %d", showgrid=False),
                yaxis=dict(title="Predicted Share", range=[0, 1.05],
                           showgrid=True, gridcolor='rgba(100,116,139,0.15)'),
                height=300,
                margin=dict(l=10, r=10, t=40, b=30),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        elif n_dates == 1:
            st.info("Only one data snapshot so far. Run `extract_current.py` again later to see the trend over time.")

    # ── Key stats table with row selection ────────────────────────────────────
    st.markdown("##### Key Stats")

    with st.expander("📖 What do these columns mean?"):
        st.markdown("""
| Column | What it means |
|---|---|
| **PTS** | Points per game |
| **W/L%** | Team's win percentage — MVP voters heavily favor players on winning teams |
| **WS/48** | Win Shares per 48 minutes — estimates how many wins a player contributes per game played. Higher = more valuable |
| **VORP** | Value Over Replacement Player — how much better the player is vs. a replacement-level player. Higher = more dominant |
| **OBPM** | Offensive Box Plus-Minus — offensive contribution per 100 possessions vs. the league average. Positive = above average |
| **Seed** | Conference playoff seeding (1 = best record in conference, 15 = worst). MVP candidates almost always come from top-seeded teams |
| **Pred. Share** | The model's predicted fraction of total MVP votes (0–1). This is what the model is optimizing |
""")

    stats_cols = ['Player', 'PTS', 'W/L%', 'win_shares_per_48_minutes',
                  'value_over_replacement_player', 'offensive_box_plus_minus', 'seed', 'predictions']
    available_cols = [c for c in stats_cols if c in display_df.columns]
    stats_display = (display_df[available_cols]
                     .rename(columns={'win_shares_per_48_minutes': 'WS/48',
                                      'value_over_replacement_player': 'VORP',
                                      'offensive_box_plus_minus': 'OBPM',
                                      'predictions': 'Pred. Share'})
                     .reset_index(drop=True))
    if 'Pred. Share' in stats_display.columns:
        stats_display['Pred. Share'] = stats_display['Pred. Share'].round(3)

    event = st.dataframe(stats_display, use_container_width=True,
                         on_select="rerun", selection_mode="single-row",
                         height=min(280, 42 + len(stats_display) * 37))

    # ── Deep dive ──────────────────────────────────────────────────────────────
    selected_rows = event.selection.rows
    if selected_rows:
        selected_player = display_df.iloc[selected_rows[0]]['Player']
        player_row = all_candidates[all_candidates['Player'] == selected_player].iloc[0]

        st.markdown("---")

        headshot_url = cached_headshot(selected_player)
        X_all = all_candidates[feat_cols].values
        player_idx = all_candidates[all_candidates['Player'] == selected_player].index[0]

        # ── Header: photo + name + pred badge ─────────────────────────────────
        col_photo, col_header = st.columns([1, 5])
        with col_photo:
            if headshot_url:
                st.image(headshot_url, width=120)
        with col_header:
            team = player_row.get('Team', '')
            pred = float(player_row.get('predictions', 0))
            seed_val = player_row.get('seed', None)
            try:
                seed_str = f"#{int(float(seed_val))} seed"
            except (TypeError, ValueError):
                seed_str = ""
            st.markdown(
                f"<div style='padding-top:10px'>"
                f"<span style='font-size:1.5rem;font-weight:700'>{selected_player}</span>"
                f"<span style='font-size:1rem;color:#94a3b8;margin-left:10px'>{team}"
                f"{' · ' + seed_str if seed_str else ''}</span>"
                f"</div>"
                f"<div style='margin-top:8px'>"
                f"<span style='background:#1d4ed8;color:#fff;padding:4px 14px;"
                f"border-radius:20px;font-size:0.9rem;font-weight:600'>"
                f"Predicted Vote Share: {pred:.3f}</span>"
                f"</div>"
                f"<p style='color:#64748b;font-size:0.8rem;margin-top:8px'>"
                f"The model estimates this player would receive {pred*100:.1f}% of total MVP votes "
                f"based on their current stats.</p>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Key stats as metric cards ──────────────────────────────────────────
        def safe_get(col, fmt="{:.1f}"):
            val = player_row.get(col, None)
            try:
                return fmt.format(float(val))
            except (TypeError, ValueError):
                return "—"

        mc = st.columns(6)
        mc[0].metric("PTS / Game", safe_get('PTS', "{:.1f}"))
        mc[1].metric("Team W/L%", safe_get('W/L%', "{:.3f}"))
        mc[2].metric("WS / 48 min", safe_get('win_shares_per_48_minutes', "{:.3f}"))
        mc[3].metric("VORP", safe_get('value_over_replacement_player', "{:.1f}"))
        mc[4].metric("Off. BPM", safe_get('offensive_box_plus_minus', "{:.1f}"))
        try:
            seed_display = f"#{int(float(player_row.get('seed', 0)))}"
        except (TypeError, ValueError):
            seed_display = "—"
        mc[5].metric("Conf. Seed", seed_display)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # ── SHAP force plot ────────────────────────────────────────────────────
        st.markdown(
            "<p style='font-size:0.8rem;color:#94a3b8;text-transform:uppercase;"
            "letter-spacing:0.07em;margin-bottom:2px'>Why this prediction?</p>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Red arrows push the prediction higher (toward more MVP votes). "
            "Blue arrows push it lower. Longer arrows = bigger impact on the final score."
        )
        if best_model is not None:
            html_plot = generate_shap_force_plots(best_model, X_all, [player_idx], feat_cols)[0]
            st.components.v1.html(html_plot, height=175, scrolling=False)

        # ── A5: What-If Sliders ────────────────────────────────────────────────
        if best_model is not None and df is not None:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            with st.expander("🔮 What-If Analysis — Adjust this player's stats"):
                st.caption(
                    "Slide any stat to see how a change would affect the predicted vote share. "
                    "All other stats stay fixed."
                )
                slider_features = [
                    ('PTS', 'Points / Game', 0.0, 50.0, 0.1),
                    ('W/L%', 'Team W/L%', 0.20, 0.90, 0.01),
                    ('win_shares_per_48_minutes', 'WS / 48 min', 0.0, 0.35, 0.005),
                    ('value_over_replacement_player', 'VORP', -3.0, 15.0, 0.1),
                    ('offensive_box_plus_minus', 'Off. BPM', -5.0, 20.0, 0.1),
                ]
                available_sliders = [(col, label, lo, hi, step)
                                     for col, label, lo, hi, step in slider_features
                                     if col in feat_cols]

                modified_vals = {}
                slider_cols = st.columns(len(available_sliders))
                for i, (col, label, lo, hi, step) in enumerate(available_sliders):
                    try:
                        current = float(player_row.get(col, 0) or 0)
                        current = max(lo, min(hi, current))
                    except (TypeError, ValueError):
                        current = (lo + hi) / 2
                    with slider_cols[i]:
                        modified_vals[col] = st.slider(
                            label, min_value=lo, max_value=hi, value=current, step=step,
                            key=f"whatif_{col}_{selected_player}",
                        )

                # Build modified feature vector
                mod_vec = []
                for c in feat_cols:
                    if c in modified_vals:
                        mod_vec.append(modified_vals[c])
                    else:
                        try:
                            mod_vec.append(float(player_row.get(c, 0) or 0))
                        except (TypeError, ValueError):
                            mod_vec.append(0.0)

                new_pred = float(best_model.predict(np.array([mod_vec]))[0])
                delta = new_pred - pred
                st.metric(
                    "New Predicted Vote Share",
                    f"{new_pred:.3f}",
                    delta=f"{delta:+.3f}",
                    delta_color="normal" if delta >= 0 else "inverse",
                )

        # ── A3: Historical Comps ───────────────────────────────────────────────
        if df is not None and feat_cols:
            similar = find_similar_seasons(player_row, df, feat_cols, n=3)
            if similar:
                st.markdown("---")
                st.markdown("##### Most Similar Historical MVP Seasons")
                st.caption(
                    f"The 3 past MVP-winning seasons whose stats most closely match "
                    f"{selected_player}'s current numbers (normalized Euclidean distance on model features)."
                )
                stat_label_map = [
                    ('PTS',                          'PTS',    '{:.1f}'),
                    ('W/L%',                         'Win%',   '{:.3f}'),
                    ('win_shares_per_48_minutes',    'WS/48',  '{:.3f}'),
                    ('value_over_replacement_player','VORP',   '{:.1f}'),
                    ('offensive_box_plus_minus',     'OBPM',   '{:.1f}'),
                    ('seed',                         'Seed',   '#{:.0f}'),
                ]
                col_range_cache = {col: (df[col].max() - df[col].min())
                                   for col, _, _ in stat_label_map if col in df.columns}

                comp_cols = st.columns(3)
                for i, comp in enumerate(similar):
                    with comp_cols[i]:
                        comp_headshot = cached_headshot(comp['Player'])
                        if comp_headshot:
                            st.image(comp_headshot, width=80)
                        similarity_pct = comp['similarity'] * 100

                        # Build reasons: top-2 closest stats
                        comp_hist_row = df[(df['Player'] == comp['Player']) &
                                          (df['year'] == comp['year'])]
                        reason_lines = []
                        if not comp_hist_row.empty:
                            comp_hist = comp_hist_row.iloc[0]
                            diffs = []
                            for col, label, fmt in stat_label_map:
                                if col not in feat_cols:
                                    continue
                                try:
                                    cv = float(player_row.get(col, 0) or 0)
                                    hv = float(comp_hist.get(col, 0) or 0)
                                    rng = col_range_cache.get(col, 1) or 1
                                    diffs.append((abs(cv - hv) / rng, label, fmt, cv, hv))
                                except (TypeError, ValueError):
                                    pass
                            diffs.sort()
                            for _, label, fmt, cv, hv in diffs[:3]:
                                reason_lines.append(
                                    f"{label}: {fmt.format(cv)} vs {fmt.format(hv)}"
                                )

                        reasons_html = ''.join(
                            f"<span style='color:#64748b;font-size:0.75rem'>{r}</span><br>"
                            for r in reason_lines
                        )
                        st.markdown(
                            f"<div style='margin-top:4px'>"
                            f"<span style='font-weight:700'>{comp['Player']}</span><br>"
                            f"<span style='color:#94a3b8;font-size:0.85rem'>{comp['year']} · "
                            f"Share: {comp['Share']:.3f}</span><br>"
                            f"<span style='color:#60a5fa;font-size:0.8rem'>"
                            f"Similarity: {similarity_pct:.0f}%</span><br>"
                            f"{reasons_html}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
    else:
        st.markdown(
            "<p style='color:#475569;font-size:0.85rem;margin-top:8px'>"
            "↑ Click a row in the table to see a full breakdown for that player.</p>",
            unsafe_allow_html=True,
        )
