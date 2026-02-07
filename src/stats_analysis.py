"""
Statistical profitability analysis for the NFL spread model.

Answers the question: "Is this model profitable, or is the apparent edge just noise?"
"""
import numpy as np
import pandas as pd
from scipy import stats
from odds import american_to_implied_prob, american_to_decimal


# ---------------------------------------------------------------------------
# 1. Bootstrap ROI Confidence Intervals
# ---------------------------------------------------------------------------
def bootstrap_roi_ci(pnl_series, n_bootstrap=10000, ci=0.95, stake_unit=1.0):
    """
    Bootstrap confidence interval for ROI per bet.

    Uses resampling (no normality assumption) because bet PnL is bimodal
    (you either win +0.909 or lose -1.0 at -110).
    """
    pnl = pnl_series.values
    n = len(pnl)

    rng = np.random.default_rng(42)
    boot_rois = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(pnl, size=n, replace=True)
        boot_rois[i] = sample.mean() / stake_unit

    alpha = (1 - ci) / 2
    ci_lower = float(np.percentile(boot_rois, alpha * 100))
    ci_upper = float(np.percentile(boot_rois, (1 - alpha) * 100))

    # One-sided p-value: P(ROI <= 0)
    p_value = float(np.mean(boot_rois <= 0))

    return {
        "mean_roi": float(np.mean(boot_rois)),
        "median_roi": float(np.median(boot_rois)),
        "std_roi": float(np.std(boot_rois)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci,
        "p_value_positive": p_value,
        "n_bets": n,
        "n_bootstrap": n_bootstrap,
    }


# ---------------------------------------------------------------------------
# 2. Binomial Significance Test
# ---------------------------------------------------------------------------
def binomial_significance_test(n_wins, n_bets, breakeven_prob=0.5238):
    """
    Test whether the observed hit rate is significantly above the break-even rate.

    For -110 odds, break-even is 52.38%.
    Uses exact binomial test (scipy.stats.binomtest).
    """
    result = stats.binomtest(n_wins, n_bets, breakeven_prob, alternative='greater')
    ci = result.proportion_ci(confidence_level=0.95)

    return {
        "n_wins": int(n_wins),
        "n_bets": int(n_bets),
        "observed_rate": n_wins / n_bets if n_bets > 0 else 0,
        "breakeven_rate": breakeven_prob,
        "p_value": float(result.pvalue),
        "is_significant_5pct": bool(result.pvalue < 0.05),
        "is_significant_1pct": bool(result.pvalue < 0.01),
        "ci_95_lower": float(ci.low),
        "ci_95_upper": float(ci.high),
    }


# ---------------------------------------------------------------------------
# 3. Risk-Adjusted Return Metrics
# ---------------------------------------------------------------------------
def risk_metrics(pnl_series):
    """
    Sharpe ratio, Sortino ratio, max drawdown, max drawdown duration,
    and profit factor.
    """
    returns = pnl_series.values
    n = len(returns)

    if n < 2 or np.std(returns) == 0:
        return {
            "sharpe": 0, "sortino": 0, "max_drawdown": 0,
            "max_dd_duration": 0, "profit_factor": 0,
            "total_pnl": float(returns.sum()) if n > 0 else 0,
            "n_bets": n, "win_rate": 0,
        }

    mean_ret = np.mean(returns)
    sharpe = float(mean_ret / np.std(returns, ddof=1))

    # Sortino (only penalize downside)
    downside = returns[returns < 0]
    if len(downside) > 1:
        downside_std = np.std(downside, ddof=1)
    else:
        downside_std = np.std(returns, ddof=1)
    sortino = float(mean_ret / downside_std) if downside_std > 0 else 0.0

    # Max Drawdown
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    max_dd = float(drawdown.min())

    # Max Drawdown Duration (bets between peak and recovery)
    in_dd = drawdown < 0
    max_dd_dur = 0
    current_dur = 0
    for v in in_dd:
        if v:
            current_dur += 1
            max_dd_dur = max(max_dd_dur, current_dur)
        else:
            current_dur = 0

    # Profit Factor (gross wins / gross losses)
    gross_wins = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    profit_factor = float(gross_wins / gross_losses) if gross_losses > 0 else float('inf')

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "max_dd_duration": max_dd_dur,
        "profit_factor": profit_factor,
        "total_pnl": float(returns.sum()),
        "n_bets": n,
        "win_rate": float((returns > 0).mean()),
    }


# ---------------------------------------------------------------------------
# 4. Closing Line Value (CLV) Analysis
# ---------------------------------------------------------------------------
def clv_analysis(bets_df):
    """
    CLV measures whether the model systematically gets a better price
    than the closing line. This is the gold standard for evaluating
    sports betting models.

    Requires a 'closing_odds' column in bets_df. Returns a stub message
    if that column is absent.
    """
    if "closing_odds" not in bets_df.columns:
        return {
            "available": False,
            "note": ("Closing line data not available. "
                     "Add a 'closing_odds' column to enable CLV analysis.")
        }

    df = bets_df.copy()
    df["implied_opening"] = df["odds_home"].apply(american_to_implied_prob)
    df["implied_closing"] = df["closing_odds"].apply(american_to_implied_prob)
    df["clv"] = df["implied_closing"] - df["implied_opening"]

    return {
        "available": True,
        "mean_clv": float(df["clv"].mean()),
        "median_clv": float(df["clv"].median()),
        "pct_positive_clv": float((df["clv"] > 0).mean()),
        "clv_ttest_pvalue": float(stats.ttest_1samp(df["clv"], 0).pvalue),
    }


# ---------------------------------------------------------------------------
# 5. Season-by-Season Breakdown
# ---------------------------------------------------------------------------
def season_breakdown(bets_df):
    """Group bets by season and compute per-season stats."""
    if "season" not in bets_df.columns or bets_df.empty:
        return pd.DataFrame()

    def _season_stats(group):
        n = len(group)
        wins = int(group["won"].sum())
        pnl = group["pnl"].sum()
        cumulative = group["pnl"].cumsum()
        peak = cumulative.cummax()
        max_dd = float((cumulative - peak).min())
        return pd.Series({
            "n_bets": n,
            "wins": wins,
            "losses": n - wins,
            "hit_rate": round(wins / n, 4) if n > 0 else 0,
            "total_pnl": round(pnl, 2),
            "roi_per_bet": round(pnl / n, 4) if n > 0 else 0,
            "max_drawdown": round(max_dd, 2),
        })

    return bets_df.groupby("season").apply(_season_stats).reset_index()


# ---------------------------------------------------------------------------
# 6. Monte Carlo Null Distribution
# ---------------------------------------------------------------------------
def monte_carlo_null_distribution(n_bets, win_prob=0.5238, odds_vig=-110,
                                  n_simulations=10000, seed=42):
    """
    Simulate random betting to calibrate expectations.

    Answers: "If I randomly bet at -110 odds, what PnL distribution
    would I see over N bets?"
    """
    rng = np.random.default_rng(seed)
    dec_odds = american_to_decimal(odds_vig)

    sim_pnls = np.empty(n_simulations)
    for i in range(n_simulations):
        wins = rng.binomial(n_bets, win_prob)
        losses = n_bets - wins
        sim_pnls[i] = wins * (dec_odds - 1) - losses * 1.0

    return {
        "sim_pnls": sim_pnls,
        "mean_pnl": float(np.mean(sim_pnls)),
        "std_pnl": float(np.std(sim_pnls)),
        "percentile_5": float(np.percentile(sim_pnls, 5)),
        "percentile_95": float(np.percentile(sim_pnls, 95)),
    }


def observed_vs_null_pvalue(observed_pnl, null_distribution):
    """
    What fraction of random simulations produced PnL >= observed?
    Low p-value means the result is unlikely under random betting.
    """
    return float(np.mean(null_distribution["sim_pnls"] >= observed_pnl))


# ---------------------------------------------------------------------------
# 7. Comprehensive Report Generator
# ---------------------------------------------------------------------------
def generate_profitability_report(bets_df, stake_unit=1.0, odds_vig=-110):
    """
    Master function: runs all analyses and returns a structured report
    with a clear verdict.
    """
    report = {}

    if bets_df.empty:
        report["error"] = "No bets to analyze."
        return report

    n_bets = len(bets_df)
    n_wins = int(bets_df["won"].sum())

    # 1. Basic stats
    report["basic"] = {
        "total_bets": n_bets,
        "total_wins": n_wins,
        "total_losses": n_bets - n_wins,
        "hit_rate": round(n_wins / n_bets, 4),
        "total_pnl": round(float(bets_df["pnl"].sum()), 2),
        "roi_per_bet": round(float(bets_df["pnl"].mean() / stake_unit), 4),
    }

    # 2. Bootstrap CI
    report["bootstrap"] = bootstrap_roi_ci(bets_df["pnl"], stake_unit=stake_unit)

    # 3. Binomial test
    breakeven = american_to_implied_prob(odds_vig)
    report["binomial_test"] = binomial_significance_test(n_wins, n_bets, breakeven)

    # 4. Risk metrics
    report["risk"] = risk_metrics(bets_df["pnl"])

    # 5. CLV
    report["clv"] = clv_analysis(bets_df)

    # 6. Season breakdown
    report["by_season"] = season_breakdown(bets_df)

    # 7. Monte Carlo
    null_dist = monte_carlo_null_distribution(n_bets, win_prob=breakeven, odds_vig=odds_vig)
    observed_pnl = float(bets_df["pnl"].sum())
    report["monte_carlo"] = {
        "null_mean_pnl": null_dist["mean_pnl"],
        "null_std_pnl": null_dist["std_pnl"],
        "observed_pnl": observed_pnl,
        "p_value": observed_vs_null_pvalue(observed_pnl, null_dist),
    }

    # 8. Verdict
    p_bootstrap = report["bootstrap"]["p_value_positive"]
    p_binomial = report["binomial_test"]["p_value"]
    p_mc = report["monte_carlo"]["p_value"]

    if p_bootstrap < 0.05 and p_binomial < 0.05:
        verdict = "STATISTICALLY SIGNIFICANT EDGE DETECTED"
        recommendation = "Consider increasing bet volume and monitoring CLV for confirmation."
    elif p_bootstrap < 0.10 or p_binomial < 0.10:
        verdict = "WEAK EVIDENCE OF EDGE -- MORE DATA NEEDED"
        recommendation = ("Results are suggestive but not conclusive. "
                          "Continue collecting out-of-sample data before scaling up.")
    else:
        verdict = "NO SIGNIFICANT EDGE DETECTED -- RESULTS CONSISTENT WITH RANDOM NOISE"
        recommendation = ("The model needs significantly more predictive features or a "
                          "fundamentally different approach to overcome the vig.")

    report["verdict"] = {
        "conclusion": verdict,
        "p_bootstrap": p_bootstrap,
        "p_binomial": p_binomial,
        "p_monte_carlo": p_mc,
        "recommendation": recommendation,
    }

    return report
