import pandas as pd
from odds import american_to_implied_prob, american_to_decimal


def backtest_spread(df, min_edge=0.01, stake_unit=1.0, odds_vig=-110):
    """
    Backtests the model's predictions against historical outcomes.
    Evaluates both home-side and away-side bets.
    """
    df = df.copy()

    # Filter out pushes (home_cover is NaN)
    df = df.dropna(subset=["home_cover"])

    # Validation
    required_cols = ["date", "home_team", "away_team", "home_cover", "spread_home"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for backtest: {missing}")

    if "p_home" not in df.columns:
        print("Warning: 'p_home' column not found. Cannot calculate edge.")
        return pd.DataFrame(), {"bets": 0, "hit_rate": 0, "total_pnl": 0, "roi_per_bet": 0}

    implied = american_to_implied_prob(odds_vig)
    dec_odds = american_to_decimal(odds_vig)

    rows = []
    for _, row in df.iterrows():
        p_home = row["p_home"]
        p_away = 1 - p_home
        edge_home = p_home - implied
        edge_away = p_away - implied

        # Home-side bet
        if edge_home >= min_edge:
            won = int(row["home_cover"] == 1)
            pnl = stake_unit * (dec_odds - 1) if won else -stake_unit
            rows.append({
                "date": row["date"], "season": row.get("season"),
                "week": row.get("week"),
                "home_team": row["home_team"], "away_team": row["away_team"],
                "spread_home": row["spread_home"],
                "side": "home", "p_model": p_home, "p_implied": implied,
                "edge": edge_home, "won": won, "pnl": pnl
            })

        # Away-side bet
        if edge_away >= min_edge:
            won = int(row["home_cover"] == 0)
            pnl = stake_unit * (dec_odds - 1) if won else -stake_unit
            rows.append({
                "date": row["date"], "season": row.get("season"),
                "week": row.get("week"),
                "home_team": row["home_team"], "away_team": row["away_team"],
                "spread_home": row["spread_home"],
                "side": "away", "p_model": p_away, "p_implied": implied,
                "edge": edge_away, "won": won, "pnl": pnl
            })

    bets = pd.DataFrame(rows)
    if bets.empty:
        return bets, {"bets": 0, "hit_rate": 0, "total_pnl": 0, "roi_per_bet": 0}

    bets = bets.sort_values("date")
    bets["bankroll"] = bets["pnl"].cumsum()

    summary = {
        "bets": len(bets),
        "hit_rate": bets["won"].mean(),
        "total_pnl": bets["pnl"].sum(),
        "roi_per_bet": bets["pnl"].mean() / stake_unit,
        "max_drawdown": _max_drawdown(bets["bankroll"]),
    }
    return bets, summary


def _max_drawdown(equity_curve):
    """Compute maximum peak-to-trough drawdown from a cumulative PnL series."""
    peak = equity_curve.cummax()
    drawdown = equity_curve - peak
    return drawdown.min()


def edge_sensitivity_analysis(df, thresholds=None, stake_unit=1.0, odds_vig=-110):
    """
    Run the backtest at multiple edge thresholds to see
    how volume vs. profitability trade off.
    """
    if thresholds is None:
        thresholds = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]

    results = []
    for t in thresholds:
        _, summary = backtest_spread(df, min_edge=t, stake_unit=stake_unit, odds_vig=odds_vig)
        summary["min_edge_threshold"] = t
        results.append(summary)

    return pd.DataFrame(results)
