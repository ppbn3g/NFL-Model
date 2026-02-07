import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx
import dash_bootstrap_components as dbc

# --- PATH SETUP ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path: sys.path.insert(0, SRC)

# --- IMPORTS ---
try:
    from todays_picks import get_todays_picks
    from train import load_nflfastR_games
    from features import add_targets, build_rolling_team_features, DEFAULT_ROLLING_WINDOW
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    # Fallbacks to prevent crash
    def get_todays_picks(min_edge=0.0): return pd.DataFrame()
    def load_nflfastR_games(): return pd.DataFrame()
    def add_targets(df): return df
    def build_rolling_team_features(df, w): return df
    DEFAULT_ROLLING_WINDOW = 10

# --- APP SETUP ---
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "NFL Command Center"

# --- DATA LOADER ---
print("Initializing Data...")
try:
    _HIST_DF = load_nflfastR_games()
    _HIST_DF = add_targets(_HIST_DF)
    _HIST_DF = build_rolling_team_features(_HIST_DF, window=DEFAULT_ROLLING_WINDOW)
    HIST_DATA = _HIST_DF.dropna(subset=["home_team", "away_team", "spread_home", "home_cover"]).copy()
    
    # Calculate ATS Records for the Leaderboard
    def get_ats_records(df):
        # Home Covers
        h_covers = df[df['home_cover'] == 1]['home_team'].value_counts()
        h_games = df['home_team'].value_counts()
        
        # Away Covers (home_cover == 0 means away covered)
        a_covers = df[df['home_cover'] == 0]['away_team'].value_counts()
        a_games = df['away_team'].value_counts()
        
        total_covers = h_covers.add(a_covers, fill_value=0)
        total_games = h_games.add(a_games, fill_value=0)
        
        ats_df = pd.DataFrame({'Covers': total_covers, 'Games': total_games})
        ats_df['Cover %'] = (ats_df['Covers'] / ats_df['Games'])
        ats_df = ats_df.sort_values('Cover %', ascending=False).reset_index().rename(columns={'index': 'Team'})
        ats_df['Cover %'] = ats_df['Cover %'].apply(lambda x: f"{x:.1%}")
        return ats_df

    ATS_STATS = get_ats_records(HIST_DATA[HIST_DATA['season'] == HIST_DATA['season'].max()])

except Exception as e:
    print(f"Data Warning: {e}")
    HIST_DATA = pd.DataFrame()
    ATS_STATS = pd.DataFrame(columns=["Team", "Covers", "Games", "Cover %"])

# --- HELPER: KELLY CRITERION ---
def calculate_kelly(odds, p_model, bankroll):
    decimal_odds = (100 / (odds + 100)) + 1 if odds > 0 else (1 + (100 / abs(odds)))
    b = decimal_odds - 1
    p = p_model
    q = 1 - p
    f = (b * p - q) / b
    return max(0, (f * 0.25) * bankroll) # 1/4 Kelly for safety

# --- LAYOUT ---
app.layout = dbc.Container([
    # NAVBAR
    dbc.NavbarSimple(
        brand="🏈 NFL Edge Command Center",
        brand_href="#",
        color="dark",
        dark=True,
        className="mb-3 border-bottom border-secondary",
    ),

    dbc.Row([
        # ==========================
        # LEFT SIDEBAR
        # ==========================
        dbc.Col([
            # 1. BANKROLL
            dbc.Card([
                dbc.CardHeader("💰 Bankroll"),
                dbc.CardBody([
                    dbc.InputGroup([
                        dbc.InputGroupText("$"),
                        dbc.Input(id="input-bankroll", type="number", value=1000)
                    ], className="mb-2"),
                    html.Small("Adjusts suggested bet sizes", className="text-muted")
                ])
            ], className="mb-3 shadow-sm"),

            # 2. FILTERS
            dbc.Card([
                dbc.CardHeader("🔍 Market Filters"),
                dbc.CardBody([
                    html.Label("Teams"),
                    dcc.Dropdown(id="team-filter", placeholder="Select...", multi=True, style={"color": "black"}),
                    html.Br(),
                    html.Label("Books"),
                    dcc.Dropdown(id="book-filter", placeholder="All Books", multi=True, style={"color": "black"}),
                    html.Br(),
                    html.Label("Min Edge"),
                    dcc.Slider(id="edge-slider", min=0, max=0.05, step=0.01, value=0.0, marks={0:'0%', 0.05:'5%'}),
                    dbc.Button("Refresh Odds", id="btn-refresh", color="primary", className="mt-3 w-100")
                ])
            ], className="mb-3 shadow-sm"),

            # 3. BET SLIP
            dbc.Card([
                dbc.CardHeader("📝 Active Bet Slip"),
                dbc.CardBody([
                    html.Div(id="bet-slip-content"),
                    html.Hr(),
                    html.H5(id="slip-total-ev", className="text-success m-0")
                ])
            ], className="shadow-sm border-info")
        ], width=3),

        # ==========================
        # RIGHT MAIN PANEL
        # ==========================
        dbc.Col([
            # TABS: LIVE GAMES vs LEAGUE INTEL
            dbc.Tabs([
                # TAB 1: LIVE GAMES
                dbc.Tab(label="📡 Live Market", children=[
                    dbc.Card([
                        dbc.CardBody([
                            dash_table.DataTable(
                                id="live-table",
                                sort_action="native",
                                row_selectable="multi",
                                selected_rows=[],
                                style_as_list_view=True,
                                style_header={'backgroundColor': '#222', 'color': 'white', 'fontWeight': 'bold'},
                                style_cell={'backgroundColor': '#333', 'color': 'white', 'padding': '12px'},
                                style_data_conditional=[
                                    {'if': {'filter_query': '{Edge} >= 0.02', 'column_id': 'Edge'}, 'backgroundColor': '#198754', 'color': 'white'},
                                    {'if': {'state': 'selected'}, 'backgroundColor': '#0dcaf0', 'color': 'black', 'border': '1px solid white'}
                                ]
                            )
                        ], className="p-0")
                    ], className="mb-4 shadow-sm border-0"),
                ]),

                # TAB 2: LEAGUE INTEL (ATS STANDINGS)
                dbc.Tab(label="🏆 ATS Leaderboard", children=[
                    dbc.Card([
                        dbc.CardBody([
                            dash_table.DataTable(
                                data=ATS_STATS.to_dict('records'),
                                columns=[{"name": i, "id": i} for i in ATS_STATS.columns],
                                sort_action="native",
                                style_as_list_view=True,
                                style_header={'backgroundColor': '#222', 'color': 'white'},
                                style_cell={'backgroundColor': '#333', 'color': 'white', 'padding': '10px'},
                            )
                        ])
                    ], className="mb-4 shadow-sm border-0")
                ])
            ], className="mb-3"),

            # MATCHUP ANALYSIS PANEL (Changes based on selection)
            dbc.Card([
                dbc.CardHeader("📊 Matchup Intelligence"),
                dbc.CardBody([
                    dbc.Row([
                        # GAUGE CHART
                        dbc.Col(dcc.Graph(id="chart-gauge", style={"height": "250px"}), width=4),
                        # HISTORY & SCORING
                        dbc.Col([
                            html.H6("Recent History (Head-to-Head)", className="text-info"),
                            html.Div(id="h2h-content"),
                            html.Hr(),
                            dcc.Graph(id="chart-trend", style={"height": "200px"})
                        ], width=8)
                    ])
                ])
            ], className="shadow-sm")

        ], width=9)
    ])
], fluid=True, className="p-4")


# ==========================
# CALLBACKS
# ==========================

# 1. LOAD TABLE
@app.callback(
    Output("team-filter", "options"),
    Output("book-filter", "options"),
    Output("live-table", "data"),
    Output("live-table", "columns"),
    Input("btn-refresh", "n_clicks"),
    Input("team-filter", "value"),
    Input("book-filter", "value"),
    Input("edge-slider", "value")
)
def update_table(n_clicks, teams, books, min_edge):
    try:
        df = get_todays_picks(min_edge=0.0)
        if df.empty: return [], [], [], []

        # Filters
        all_teams = sorted(list(set(df["home_team"].unique()) | set(df["away_team"].unique())))
        team_opts = [{"label": t, "value": t} for t in all_teams]
        all_books = sorted(df["book"].unique())
        book_opts = [{"label": b, "value": b} for b in all_books]

        mask = (df["edge"] >= float(min_edge))
        if teams: mask = mask & (df["home_team"].isin(teams) | df["away_team"].isin(teams))
        if books: mask = mask & (df["book"].isin(books))
        
        filtered = df[mask].copy()
        
        # Display Columns
        filtered["Edge"] = filtered["edge"].apply(lambda x: f"{x:.1%}")
        filtered["Pick"] = filtered.apply(lambda r: f"BET {r['home_team']}" if r['bet'] else "PASS", axis=1)
        
        # Add Date if exists
        if "commence_time" in filtered.columns:
            filtered["Time"] = pd.to_datetime(filtered["commence_time"]).dt.strftime("%m-%d %H:%M")
        else:
            filtered["Time"] = "N/A"

        cols = ["Time", "away_team", "home_team", "book", "spread_home", "odds_home", "Edge", "Pick"]
        columns = [{"name": c, "id": c} for c in cols]
        
        return team_opts, book_opts, filtered.to_dict("records"), columns
    except Exception as e:
        print(e)
        return [], [], [], []

# 2. UPDATE ANALYSIS & SLIP
@app.callback(
    Output("bet-slip-content", "children"),
    Output("slip-total-ev", "children"),
    Output("chart-gauge", "figure"),
    Output("h2h-content", "children"),
    Output("chart-trend", "figure"),
    Input("live-table", "selected_rows"),
    Input("live-table", "data"),
    State("input-bankroll", "value")
)
def update_analysis(selected_indices, table_data, bankroll):
    # Default State
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    if not table_data or not selected_indices:
        return "Select games to bet.", "$0.00 EV", empty_fig, "Select a game to see history.", empty_fig

    selected_rows = [table_data[i] for i in selected_indices]
    
    # --- BUILD BET SLIP ---
    slip_items = []
    total_ev = 0
    for row in selected_rows:
        try:
            edge_val = float(row["Edge"].strip('%')) / 100
            odds = row['odds_home']
            # Re-calc Model Prob from Edge + Implied
            implied = 100/(odds+100) if odds > 0 else (-odds)/(-odds+100)
            p_model = implied + edge_val
            
            wager = calculate_kelly(odds, p_model, bankroll or 1000)
            ev = wager * edge_val
            total_ev += ev
            
            slip_items.append(dbc.ListGroupItem([
                html.Div([
                    html.Strong(f"{row['away_team']} @ {row['home_team']}"),
                    html.Span(f" ${wager:.0f}", className="float-end text-info")
                ]),
                html.Small(f"{row['book']} ({row['odds_home']}) | Edge: {row['Edge']}", className="text-muted")
            ], color="dark"))
        except (ValueError, KeyError, TypeError) as e:
            print(f"Bet slip calculation error: {e}")

    # --- ANALYZE LAST SELECTED GAME ---
    last_game = selected_rows[-1]
    home = last_game['home_team']
    away = last_game['away_team']
    
    # 1. Gauge Chart
    try:
        edge_raw = float(last_game["Edge"].strip('%'))
        # Normalize: 0% is Red, 5% is Max Green
        gauge_val = min(edge_raw * 100, 10) 
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = edge_raw * 100,
            title = {'text': "Model Edge %"},
            gauge = {
                'axis': {'range': [0, 6]},
                'bar': {'color': "#0dcaf0"},
                'steps': [
                    {'range': [0, 2], 'color': "#555"},
                    {'range': [2, 4], 'color': "#198754"},
                    {'range': [4, 6], 'color': "#0dcaf0"}
                ]
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", height=250, margin=dict(l=20,r=20,t=40,b=20),
                                paper_bgcolor='rgba(0,0,0,0)')
    except (ValueError, KeyError, TypeError):
        fig_gauge = empty_fig

    # 2. H2H History
    h2h_table = html.Div("No History")
    fig_trend = empty_fig
    
    if not HIST_DATA.empty:
        # Find games between these two
        mask = ((HIST_DATA['home_team'] == home) & (HIST_DATA['away_team'] == away)) | \
               ((HIST_DATA['home_team'] == away) & (HIST_DATA['away_team'] == home))
        h2h_df = HIST_DATA[mask].sort_values('date', ascending=False).head(5)
        
        if not h2h_df.empty:
            # Build mini table
            rows = []
            for _, r in h2h_df.iterrows():
                winner = r['home_team'] if r['home_score'] > r['away_score'] else r['away_team']
                rows.append(html.Tr([
                    html.Td(f"{r['season']} W{r['week']}"),
                    html.Td(f"{r['away_team']} {int(r['away_score'])} - {r['home_team']} {int(r['home_score'])}"),
                    html.Td(winner, className="text-info")
                ]))
            h2h_table = html.Table(rows, className="table table-sm table-dark")
        
        # 3. Trend Chart (Scoring)
        h_scores = HIST_DATA[HIST_DATA['home_team'] == home].sort_values('date').tail(5)
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(x=h_scores['week'], y=h_scores['home_score'], name=home, marker_color='#0dcaf0'))
        fig_trend.update_layout(template="plotly_dark", title=f"{home} Recent Scoring", height=200, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor='rgba(0,0,0,0)')

    return dbc.ListGroup(slip_items), f"Projected Profit: +${total_ev:.2f}", fig_gauge, h2h_table, fig_trend

if __name__ == "__main__":
    app.run(debug=True)