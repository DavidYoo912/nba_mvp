import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from utils import (highlight_cells, calculate_accuracy_percentage,
                   generate_shap_force_plots, prepare_evaluation_data,
                   get_headshot_url)


def display_best_model_summary(df_best_model_summary):
    # Headline accuracy metric row
    if 'Label' in df_best_model_summary.columns:
        correct = (df_best_model_summary['Label'] == 'correct').sum()
        total = len(df_best_model_summary)
        avg_mse = df_best_model_summary['MSE'].mean()
        avg_r2 = df_best_model_summary['R squared'].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MVP Winner Accuracy", f"{correct/total*100:.0f}%", f"{correct}/{total} seasons correct",
                  help="How often the model's top pick matched the actual MVP winner that season.")
        c2.metric("Seasons Evaluated", str(total),
                  help="Each season was tested on a model that had never seen that year's data.")
        c3.metric("Avg MSE", f"{avg_mse:.4f}",
                  help="Average Mean Squared Error across all seasons. Measures how far off the vote share estimates were.")
        c4.metric("Avg R²", f"{avg_r2:.3f}",
                  help="Average R-squared. Measures how well the model explains voting variation. 1.0 = perfect.")

    # ── A4: Accuracy over time bar chart ──────────────────────────────────────
    if 'Label' in df_best_model_summary.columns and 'year' in df_best_model_summary.columns:
        df_sorted = df_best_model_summary.sort_values('year').copy()
        df_sorted['year_int'] = df_sorted['year'].astype(int)
        bar_colors = ['#22c55e' if l == 'correct' else '#ef4444' for l in df_sorted['Label']]
        hover_text = [
            f"{int(r['year_int'])}<br>{'✓ Correct' if r['Label']=='correct' else '✗ Missed'}"
            f"<br>Pred: {r['Predicted MVP']}<br>Actual: {r['Actual MVP']}"
            for _, r in df_sorted.iterrows()
        ]

        fig_acc = go.Figure(go.Bar(
            x=df_sorted['year_int'],
            y=[1] * len(df_sorted),
            marker=dict(color=bar_colors, line=dict(width=0)),
            hovertext=hover_text,
            hoverinfo='text',
        ))
        fig_acc.update_layout(
            title=dict(text="Prediction Accuracy by Season  <span style='color:#22c55e'>■ Correct</span>  "
                            "<span style='color:#ef4444'>■ Missed</span>",
                       font=dict(size=13)),
            xaxis=dict(title="Season", tickmode='linear', dtick=5,
                       showgrid=False, tickfont=dict(size=11)),
            yaxis=dict(visible=False, range=[0, 1.3]),
            height=130,
            margin=dict(l=0, r=0, t=36, b=30),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            bargap=0.1,
            showlegend=False,
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    st.markdown("##### Season-by-Season Results")
    st.caption(
        "Each row is one NBA season. Green = model correctly identified the MVP winner. "
        "Red = model missed. Click any row to see the full season breakdown."
    )

    # Styling
    df_display = df_best_model_summary.copy()
    if 'year' in df_display.columns:
        df_display['year'] = df_display['year'].astype(int).astype(str)

    col_order = ['year', 'Label', 'Predicted MVP', 'Actual MVP', 'MSE', 'R squared']
    col_order = [c for c in col_order if c in df_display.columns]
    df_display = df_display[col_order]

    label_colored_cols = [c for c in ['year', 'Label', 'Predicted MVP', 'Actual MVP'] if c in df_display.columns]

    def color_by_label(row):
        color = '#22c55e' if row.get('Label') == 'correct' else '#ef4444'
        style = f'color: {color}; font-weight: 600'
        return [style if col in label_colored_cols else '' for col in row.index]

    def color_mse(val):
        c = plt.cm.RdYlGn_r(val / df_display['MSE'].max())
        return f'background-color: rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},0.15)'

    def color_r2(val):
        c = plt.cm.RdYlGn(val / df_display['R squared'].max())
        return f'background-color: rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},0.15)'

    df_styled = (df_display.reset_index(drop=True)
                 .style
                 .apply(color_by_label, axis=1)
                 .map(color_mse, subset=['MSE'])
                 .map(color_r2, subset=['R squared'])
                 .format({'MSE': '{:.4f}', 'R squared': '{:.3f}'}))

    event = st.dataframe(df_styled, use_container_width=True,
                         on_select="rerun", selection_mode="single-row",
                         height=350)

    selected_rows = event.selection.rows
    if selected_rows:
        return int(df_display.iloc[selected_rows[0]]['year'])
    return None


def display_vote_share_trend(df, player_name, df_summary=None, selected_year=None):
    """A1: Line chart of a player's MVP vote share across every season they appeared."""
    player_history = (df[df['Player'] == player_name][['year', 'Share']]
                      .dropna(subset=['Share'])
                      .groupby('year', as_index=False)['Share'].max()
                      .sort_values('year'))
    if player_history.empty:
        return

    # Determine actual MVP win years from summary
    win_years = set()
    if df_summary is not None and 'Actual MVP' in df_summary.columns and 'year' in df_summary.columns:
        wins = df_summary[df_summary['Actual MVP'] == player_name]['year']
        win_years = set(wins.astype(int).tolist())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=player_history['year'].astype(int),
        y=player_history['Share'].round(3),
        mode='lines+markers',
        name='Vote Share',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=7, color='#60a5fa'),
        hovertemplate='%{x}: %{y:.3f}<extra></extra>',
        showlegend=False,
    ))

    # Crown markers on actual MVP win years
    if win_years:
        win_rows = player_history[player_history['year'].astype(int).isin(win_years)]
        fig.add_trace(go.Scatter(
            x=win_rows['year'].astype(int),
            y=win_rows['Share'].round(3),
            mode='markers+text',
            name='MVP Winner 👑',
            text=['👑'] * len(win_rows),
            textposition='top center',
            textfont=dict(size=16),
            marker=dict(size=10, color='#f59e0b', symbol='circle'),
            hovertemplate='%{x}: MVP winner (%{y:.3f})<extra></extra>',
            showlegend=False,
        ))

    # Highlight the selected year
    if selected_year is not None:
        sel_row = player_history[player_history['year'].astype(int) == int(selected_year)]
        if not sel_row.empty:
            fig.add_trace(go.Scatter(
                x=[int(selected_year)],
                y=[round(float(sel_row.iloc[0]['Share']), 3)],
                mode='markers',
                name=f'Selected: {selected_year}',
                marker=dict(size=18, color='#f97316', symbol='circle',
                            line=dict(color='#ffffff', width=2)),
                hovertemplate=f'Selected season: {selected_year} (%{{y:.3f}})<extra></extra>',
                showlegend=True,
            ))

    fig.update_layout(
        title=dict(text=f"{player_name} — MVP Vote Share History", font=dict(size=13)),
        xaxis=dict(title="Season", tickformat='d', dtick=1, showgrid=False),
        yaxis=dict(title="Vote Share", range=[0, max(1.15, float(player_history['Share'].max()) * 1.25)],
                   showgrid=True, gridcolor='rgba(100,116,139,0.15)'),
        height=260,
        margin=dict(l=10, r=20, t=40, b=30),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=11),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='right', x=1,
            font=dict(size=10),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def evaluate_model_for_year(df, year, best_model, df_summary=None):
    df_year, X_test, feature_names, predictions, predicted_mvp, actual_mvp, mse, r2 = \
        prepare_evaluation_data(df, year, best_model)

    correct = predicted_mvp == actual_mvp

    # Metrics row with headshots
    if correct:
        # Same player — one photo, then 3 stat metrics
        c_photo, c1, c2, c3 = st.columns([1, 2, 2, 2])
        with c_photo:
            url = get_headshot_url(predicted_mvp)
            if url:
                st.image(url, width=90)
        c1.metric("Model's MVP Pick / Actual MVP", predicted_mvp,
                  delta="✓ Correct", delta_color="normal")
        c2.metric("MSE", f"{mse:.4f}",
                  help="Mean Squared Error — how far off the predicted vote shares were from actual. Lower = better.")
        c3.metric("R²", f"{r2:.3f}",
                  help="R-squared — how well the model explains the variation in voting. 1.0 = perfect, 0 = no better than guessing.")
    else:
        # Different players — two photos side by side, then stat metrics
        c_photo1, c1, c_photo2, c2, c3, c4 = st.columns([1, 2, 1, 2, 1.5, 1.5])
        with c_photo1:
            url1 = get_headshot_url(predicted_mvp)
            if url1:
                st.image(url1, width=90)
        c1.metric("Model's MVP Pick", predicted_mvp,
                  delta="✗ Missed", delta_color="off")
        with c_photo2:
            url2 = get_headshot_url(actual_mvp)
            if url2:
                st.image(url2, width=90)
        c2.metric("Actual MVP Winner", actual_mvp)
        c3.metric("MSE", f"{mse:.4f}",
                  help="Mean Squared Error — how far off the predicted vote shares were from actual. Lower = better.")
        c4.metric("R²", f"{r2:.3f}",
                  help="R-squared — how well the model explains the variation in voting. 1.0 = perfect, 0 = no better than guessing.")

    st.markdown("---")

    # ── Career MVP Vote Share History ─────────────────────────────────────────
    players_to_show = [predicted_mvp] if predicted_mvp == actual_mvp else [predicted_mvp, actual_mvp]
    st.markdown("##### Career MVP Vote Share History")
    st.caption("How this player's MVP candidacy evolved over their career. 👑 marks seasons they won the award.")
    trend_cols = st.columns(len(players_to_show))
    for i, pname in enumerate(players_to_show):
        with trend_cols[i]:
            display_vote_share_trend(df, pname, df_summary, selected_year=year)

    st.markdown("---")

    # ── Top 3 Predicted Candidates — SHAP Explanations ───────────────────────
    st.markdown("##### Top 3 Predicted Candidates — SHAP Explanations")
    st.caption(
        "Each force plot below shows *why* the model gave a player their predicted vote share. "
        "Red arrows push the prediction higher (toward more votes). Blue arrows push it lower. "
        "The length of each arrow shows how much that stat mattered for this player specifically."
    )

    candidate_names = df_year['Player'].values
    top_indices = list(np.argsort(predictions)[-3:][::-1])
    html_plots = generate_shap_force_plots(best_model, X_test, top_indices, feature_names)

    for rank, (idx, html_plot) in enumerate(zip(top_indices, html_plots), start=1):
        player_name = candidate_names[idx]
        headshot_url = get_headshot_url(player_name)

        col_photo, col_shap = st.columns([1, 6])
        with col_photo:
            if headshot_url:
                st.image(headshot_url, width=90)
            rank_label = "🥇" if rank == 1 else ("🥈" if rank == 2 else "🥉")
            st.caption(f"{rank_label} {player_name}")
        with col_shap:
            st.components.v1.html(html_plot, height=130, scrolling=False)

        if rank < 3:
            st.markdown("<div style='margin: 8px 0'></div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Feature Importance ────────────────────────────────────────────────────
    importances = best_model.feature_importances_
    feat_arr = np.array(feature_names)
    order = np.argsort(importances)

    fig = go.Figure(go.Bar(
        x=importances[order],
        y=feat_arr[order],
        orientation='h',
        marker=dict(
            color=importances[order],
            colorscale='Blues',
            showscale=False,
        ),
        text=[f"{v:.3f}" for v in importances[order]],
        textposition='outside',
    ))
    st.markdown("##### Feature Importance")
    st.caption(
        "Feature importance shows which stats the model relied on most when making predictions "
        "for this season. A longer bar means that stat had more influence on the predicted vote shares."
    )
    fig.update_layout(
        xaxis_title="Importance",
        height=max(280, len(feature_names) * 32),
        margin=dict(l=10, r=80, t=10, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
    )
    st.plotly_chart(fig, use_container_width=True)
