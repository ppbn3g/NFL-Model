"""
Comprehensive Model Evaluation & Testing Script
================================================
Tests the NFL spread model's walk-forward backtest results across every
dimension that matters for a real betting system.

Usage: python src/test_model_results.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy import stats
from odds import american_to_implied_prob, american_to_decimal

# ---------------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "walkforward_bets.csv")

def load_bets():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def print_header(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_section(title):
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# TEST 1: Overall Performance Summary
# ---------------------------------------------------------------------------
def test_overall_performance(df):
    print_header("TEST 1: OVERALL PERFORMANCE SUMMARY")

    n = len(df)
    wins = df["won"].sum()
    losses = n - wins
    hit_rate = wins / n
    total_pnl = df["pnl"].sum()
    roi = df["pnl"].mean()

    breakeven = american_to_implied_prob(-110)  # 52.38%
    edge_over_breakeven = hit_rate - breakeven

    print(f"  Total Bets:           {n}")
    print(f"  Wins / Losses:        {int(wins)} / {int(losses)}")
    print(f"  Hit Rate:             {hit_rate:.2%}")
    print(f"  Break-Even Rate:      {breakeven:.2%}")
    print(f"  Edge Over Break-Even: {edge_over_breakeven:+.2%}")
    print(f"  Total PnL:            {total_pnl:+.2f} units")
    print(f"  ROI Per Bet:          {roi:+.4f} units ({roi/1*100:+.2f}%)")
    print(f"  Avg Edge Taken:       {df['edge'].mean():.4f} ({df['edge'].mean()*100:.2f}%)")

    # Verdict
    if hit_rate > breakeven:
        print(f"\n  [PASS] Hit rate ({hit_rate:.2%}) exceeds break-even ({breakeven:.2%})")
    else:
        print(f"\n  [FAIL] Hit rate ({hit_rate:.2%}) below break-even ({breakeven:.2%})")

    if total_pnl > 0:
        print(f"  [PASS] Positive total PnL: {total_pnl:+.2f} units")
    else:
        print(f"  [FAIL] Negative total PnL: {total_pnl:+.2f} units")

    return {"hit_rate": hit_rate, "total_pnl": total_pnl, "roi": roi, "n": n}


# ---------------------------------------------------------------------------
# TEST 2: Statistical Significance Tests
# ---------------------------------------------------------------------------
def test_statistical_significance(df):
    print_header("TEST 2: STATISTICAL SIGNIFICANCE")

    n = len(df)
    wins = int(df["won"].sum())
    breakeven = american_to_implied_prob(-110)

    # Binomial Test
    print_section("Binomial Test (H0: hit_rate <= break-even)")
    binom = stats.binomtest(wins, n, breakeven, alternative='greater')
    print(f"  Observed:  {wins}/{n} = {wins/n:.2%}")
    print(f"  Expected:  {breakeven:.2%}")
    print(f"  p-value:   {binom.pvalue:.6f}")
    print(f"  Sig at 5%: {'YES' if binom.pvalue < 0.05 else 'NO'}")
    print(f"  Sig at 10%: {'YES' if binom.pvalue < 0.10 else 'NO'}")

    # Bootstrap ROI CI
    print_section("Bootstrap ROI Confidence Interval (10,000 resamples)")
    rng = np.random.default_rng(42)
    pnl = df["pnl"].values
    n_boot = 10000
    boot_rois = np.array([rng.choice(pnl, size=n, replace=True).mean() for _ in range(n_boot)])

    ci_lower = np.percentile(boot_rois, 2.5)
    ci_upper = np.percentile(boot_rois, 97.5)
    p_positive = np.mean(boot_rois <= 0)

    print(f"  Mean ROI:          {np.mean(boot_rois):+.4f}")
    print(f"  95% CI:            [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"  P(ROI <= 0):       {p_positive:.4f}")
    print(f"  P(ROI > 0):        {1 - p_positive:.4f}")

    if ci_lower > 0:
        print(f"  [PASS] 95% CI entirely above zero")
    elif ci_upper < 0:
        print(f"  [FAIL] 95% CI entirely below zero")
    else:
        print(f"  [WARN] 95% CI straddles zero -- inconclusive")

    # Monte Carlo vs Random Betting
    print_section("Monte Carlo Null Distribution (10,000 simulations)")
    dec_odds = american_to_decimal(-110)
    sim_pnls = np.empty(10000)
    rng2 = np.random.default_rng(42)
    for i in range(10000):
        sim_wins = rng2.binomial(n, breakeven)
        sim_losses = n - sim_wins
        sim_pnls[i] = sim_wins * (dec_odds - 1) - sim_losses
    observed_pnl = df["pnl"].sum()
    mc_pvalue = np.mean(sim_pnls >= observed_pnl)

    print(f"  Null Mean PnL:     {np.mean(sim_pnls):+.2f}")
    print(f"  Null Std PnL:      {np.std(sim_pnls):.2f}")
    print(f"  Observed PnL:      {observed_pnl:+.2f}")
    print(f"  MC p-value:        {mc_pvalue:.4f}")
    print(f"  Sig at 5%:         {'YES' if mc_pvalue < 0.05 else 'NO'}")

    return {
        "binom_p": binom.pvalue,
        "bootstrap_ci": (ci_lower, ci_upper),
        "mc_pvalue": mc_pvalue,
    }


# ---------------------------------------------------------------------------
# TEST 3: Risk Metrics
# ---------------------------------------------------------------------------
def test_risk_metrics(df):
    print_header("TEST 3: RISK-ADJUSTED METRICS")

    pnl = df["pnl"].values
    cumulative = np.cumsum(pnl)

    # Sharpe & Sortino
    mean_ret = np.mean(pnl)
    std_ret = np.std(pnl, ddof=1)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0

    downside = pnl[pnl < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else std_ret
    sortino = mean_ret / downside_std if downside_std > 0 else 0

    # Max Drawdown
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    max_dd = drawdown.min()
    max_dd_idx = np.argmin(drawdown)
    peak_idx = np.argmax(cumulative[:max_dd_idx + 1]) if max_dd_idx > 0 else 0

    # Max DD Duration
    in_dd = drawdown < 0
    max_dd_dur = 0
    current = 0
    for v in in_dd:
        if v:
            current += 1
            max_dd_dur = max(max_dd_dur, current)
        else:
            current = 0

    # Profit Factor
    gross_wins = pnl[pnl > 0].sum()
    gross_losses = abs(pnl[pnl < 0].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Recovery Factor
    recovery_factor = cumulative[-1] / abs(max_dd) if max_dd != 0 else float('inf')

    print(f"  Sharpe Ratio:      {sharpe:.4f}")
    print(f"  Sortino Ratio:     {sortino:.4f}")
    print(f"  Max Drawdown:      {max_dd:+.2f} units")
    print(f"  Max DD Duration:   {max_dd_dur} bets")
    print(f"  Profit Factor:     {profit_factor:.3f}")
    print(f"  Recovery Factor:   {recovery_factor:.3f}")
    print(f"  Final Bankroll:    {cumulative[-1]:+.2f} units")

    if profit_factor > 1.0:
        print(f"\n  [PASS] Profit Factor > 1.0 ({profit_factor:.3f})")
    else:
        print(f"\n  [FAIL] Profit Factor < 1.0 ({profit_factor:.3f})")

    return {"sharpe": sharpe, "sortino": sortino, "max_dd": max_dd, "profit_factor": profit_factor}


# ---------------------------------------------------------------------------
# TEST 4: Season-by-Season Breakdown
# ---------------------------------------------------------------------------
def test_season_breakdown(df):
    print_header("TEST 4: SEASON-BY-SEASON BREAKDOWN")

    seasons = sorted(df["season"].unique())
    rows = []
    for s in seasons:
        sdf = df[df["season"] == s]
        n = len(sdf)
        wins = int(sdf["won"].sum())
        pnl = sdf["pnl"].sum()
        rows.append({
            "Season": s, "Bets": n, "W": wins, "L": n - wins,
            "Hit%": f"{wins/n:.1%}", "PnL": f"{pnl:+.2f}",
            "ROI": f"{pnl/n*100:+.1f}%"
        })

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))

    winning_seasons = sum(1 for r in rows if float(r["PnL"].replace("+", "")) > 0)
    total_seasons = len(rows)
    print(f"\n  Winning Seasons:   {winning_seasons}/{total_seasons} ({winning_seasons/total_seasons:.0%})")
    print(f"  Losing Seasons:    {total_seasons - winning_seasons}/{total_seasons}")

    # Consistency check
    if winning_seasons > total_seasons / 2:
        print(f"  [PASS] Majority of seasons profitable")
    else:
        print(f"  [FAIL] Minority of seasons profitable")

    return result_df


# ---------------------------------------------------------------------------
# TEST 5: Home vs Away Bias
# ---------------------------------------------------------------------------
def test_home_away_bias(df):
    print_header("TEST 5: HOME vs AWAY SIDE ANALYSIS")

    for side in ["home", "away"]:
        sdf = df[df["side"] == side]
        n = len(sdf)
        if n == 0:
            continue
        wins = int(sdf["won"].sum())
        pnl = sdf["pnl"].sum()
        print(f"  {side.upper():>5} bets:  {n:4d}  |  Hit: {wins/n:.1%}  |  PnL: {pnl:+7.2f}  |  ROI: {pnl/n*100:+.1f}%")

    # Chi-squared test for independence
    home_df = df[df["side"] == "home"]
    away_df = df[df["side"] == "away"]
    if len(home_df) > 0 and len(away_df) > 0:
        table = pd.crosstab(df["side"], df["won"])
        chi2, p_val, _, _ = stats.chi2_contingency(table)
        print(f"\n  Chi-squared test (side vs outcome): chi2={chi2:.3f}, p={p_val:.4f}")
        if p_val < 0.05:
            print(f"  [WARN] Significant difference between home/away performance")
        else:
            print(f"  [OK] No significant home/away bias detected")


# ---------------------------------------------------------------------------
# TEST 6: Edge Size Analysis
# ---------------------------------------------------------------------------
def test_edge_analysis(df):
    print_header("TEST 6: EDGE SIZE vs PERFORMANCE")

    # Bucket by edge size
    bins = [0, 0.01, 0.02, 0.03, 0.05, 0.10, 1.0]
    labels = ["1-2%", "2-3%", "3-5%", "5-10%", "10%+"]

    df = df.copy()
    df["edge_pct"] = df["edge"] * 100
    df["edge_bucket"] = pd.cut(df["edge"], bins=bins, labels=["<1%"] + labels, right=False)

    rows = []
    for bucket in df["edge_bucket"].cat.categories:
        bdf = df[df["edge_bucket"] == bucket]
        if len(bdf) == 0:
            continue
        n = len(bdf)
        wins = int(bdf["won"].sum())
        pnl = bdf["pnl"].sum()
        rows.append({
            "Edge": bucket, "Bets": n, "Hit%": f"{wins/n:.1%}",
            "PnL": f"{pnl:+.2f}", "ROI": f"{pnl/n*100:+.1f}%"
        })

    print(pd.DataFrame(rows).to_string(index=False))

    # Correlation between edge size and outcome
    corr, p_val = stats.pointbiserialr(df["won"], df["edge"])
    print(f"\n  Point-biserial correlation (edge vs win): r={corr:.4f}, p={p_val:.4f}")
    if corr > 0 and p_val < 0.05:
        print(f"  [PASS] Higher edges correlate with more wins (as expected)")
    elif corr > 0:
        print(f"  [WEAK] Positive but not significant correlation")
    else:
        print(f"  [FAIL] No positive correlation between edge size and wins")


# ---------------------------------------------------------------------------
# TEST 7: Calibration Check
# ---------------------------------------------------------------------------
def test_calibration(df):
    print_header("TEST 7: MODEL CALIBRATION")

    df = df.copy()
    # Create probability buckets
    bins = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.65, 1.0]
    labels = ["50-52%", "52-54%", "54-56%", "56-58%", "58-60%", "60-65%", "65%+"]
    df["prob_bucket"] = pd.cut(df["p_model"], bins=bins, labels=labels, right=False)

    rows = []
    for bucket in df["prob_bucket"].cat.categories:
        bdf = df[df["prob_bucket"] == bucket]
        if len(bdf) == 0:
            continue
        n = len(bdf)
        actual = bdf["won"].mean()
        predicted = bdf["p_model"].mean()
        rows.append({
            "Predicted": bucket, "N": n,
            "Avg Pred": f"{predicted:.3f}",
            "Actual Win%": f"{actual:.3f}",
            "Gap": f"{actual - predicted:+.3f}"
        })

    print(pd.DataFrame(rows).to_string(index=False))

    # Overall Brier score
    brier = np.mean((df["p_model"] - df["won"]) ** 2)
    baseline_brier = np.mean((0.5 - df["won"]) ** 2)  # always predict 50%
    print(f"\n  Brier Score:          {brier:.6f}")
    print(f"  Baseline Brier (50%): {baseline_brier:.6f}")
    print(f"  Brier Skill Score:    {1 - brier/baseline_brier:.4f}")

    if brier < baseline_brier:
        print(f"  [PASS] Model outperforms naive 50% baseline")
    else:
        print(f"  [FAIL] Model worse than naive 50% baseline")


# ---------------------------------------------------------------------------
# TEST 8: Streak Analysis
# ---------------------------------------------------------------------------
def test_streak_analysis(df):
    print_header("TEST 8: STREAK ANALYSIS")

    results = df["won"].values

    # Find streaks
    win_streaks = []
    loss_streaks = []
    current_streak = 1

    for i in range(1, len(results)):
        if results[i] == results[i-1]:
            current_streak += 1
        else:
            if results[i-1] == 1:
                win_streaks.append(current_streak)
            else:
                loss_streaks.append(current_streak)
            current_streak = 1
    # Final streak
    if results[-1] == 1:
        win_streaks.append(current_streak)
    else:
        loss_streaks.append(current_streak)

    print(f"  Longest Win Streak:    {max(win_streaks) if win_streaks else 0}")
    print(f"  Longest Loss Streak:   {max(loss_streaks) if loss_streaks else 0}")
    print(f"  Avg Win Streak:        {np.mean(win_streaks):.1f}" if win_streaks else "")
    print(f"  Avg Loss Streak:       {np.mean(loss_streaks):.1f}" if loss_streaks else "")

    # Runs test for randomness
    n_wins = int(df["won"].sum())
    n_losses = len(df) - n_wins
    n_runs = 1 + sum(1 for i in range(1, len(results)) if results[i] != results[i-1])

    # Expected runs
    expected_runs = 1 + (2 * n_wins * n_losses) / (n_wins + n_losses)
    var_runs = (2 * n_wins * n_losses * (2 * n_wins * n_losses - n_wins - n_losses)) / \
               ((n_wins + n_losses)**2 * (n_wins + n_losses - 1))
    z_runs = (n_runs - expected_runs) / np.sqrt(var_runs)
    p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))

    print(f"\n  Runs Test (independence of outcomes):")
    print(f"    Observed Runs:   {n_runs}")
    print(f"    Expected Runs:   {expected_runs:.1f}")
    print(f"    Z-statistic:     {z_runs:.3f}")
    print(f"    p-value:         {p_runs:.4f}")
    if p_runs > 0.05:
        print(f"    [OK] Outcomes appear independent (no clustering)")
    else:
        print(f"    [WARN] Outcomes show non-random clustering")


# ---------------------------------------------------------------------------
# TEST 9: Drawdown & Recovery Analysis
# ---------------------------------------------------------------------------
def test_drawdown_analysis(df):
    print_header("TEST 9: EQUITY CURVE & DRAWDOWN ANALYSIS")

    cumulative = df["pnl"].cumsum().values
    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak

    # Find top 5 drawdowns
    dd_periods = []
    in_dd = False
    dd_start = 0
    for i in range(len(drawdown)):
        if drawdown[i] < 0 and not in_dd:
            in_dd = True
            dd_start = i
        elif drawdown[i] >= 0 and in_dd:
            in_dd = False
            dd_min = drawdown[dd_start:i].min()
            dd_periods.append((dd_start, i, dd_min, i - dd_start))
    if in_dd:
        dd_min = drawdown[dd_start:].min()
        dd_periods.append((dd_start, len(drawdown), dd_min, len(drawdown) - dd_start))

    dd_periods.sort(key=lambda x: x[2])

    print("  Top 5 Drawdown Periods:")
    print(f"  {'#':>3}  {'Depth':>8}  {'Duration':>10}  {'Start Bet':>10}  {'End Bet':>10}")
    for i, (start, end, depth, dur) in enumerate(dd_periods[:5]):
        print(f"  {i+1:>3}  {depth:>+8.2f}  {dur:>10}  {start:>10}  {end:>10}")

    # Time underwater
    underwater_pct = np.mean(drawdown < 0)
    print(f"\n  Time Underwater:    {underwater_pct:.1%} of all bets")
    print(f"  Final Equity:       {cumulative[-1]:+.2f} units")
    print(f"  Peak Equity:        {peak.max():+.2f} units")

    # Equity curve trend (simple linear regression)
    x = np.arange(len(cumulative))
    slope, intercept, r, p, _ = stats.linregress(x, cumulative)
    print(f"\n  Equity Curve Trend:")
    print(f"    Slope:  {slope:+.6f} units/bet")
    print(f"    R-sq:   {r**2:.4f}")
    print(f"    p-val:  {p:.4f}")

    if slope > 0 and p < 0.05:
        print(f"    [PASS] Significant upward trend")
    elif slope > 0:
        print(f"    [WEAK] Upward trend but not significant")
    else:
        print(f"    [FAIL] No upward trend in equity curve")


# ---------------------------------------------------------------------------
# TEST 10: Rolling Performance Windows
# ---------------------------------------------------------------------------
def test_rolling_performance(df):
    print_header("TEST 10: ROLLING PERFORMANCE (100-BET WINDOWS)")

    pnl = df["pnl"].values
    window = 100

    if len(pnl) < window:
        print("  Not enough data for rolling analysis")
        return

    rolling_pnl = pd.Series(pnl).rolling(window).sum().dropna().values
    rolling_hitrate = pd.Series(df["won"].values).rolling(window).mean().dropna().values

    breakeven = american_to_implied_prob(-110)

    print(f"  Rolling {window}-bet PnL:")
    print(f"    Best Window:     {rolling_pnl.max():+.2f} units")
    print(f"    Worst Window:    {rolling_pnl.min():+.2f} units")
    print(f"    Mean Window:     {rolling_pnl.mean():+.2f} units")
    print(f"    % Windows > 0:  {np.mean(rolling_pnl > 0):.1%}")

    print(f"\n  Rolling {window}-bet Hit Rate:")
    print(f"    Best Window:     {rolling_hitrate.max():.1%}")
    print(f"    Worst Window:    {rolling_hitrate.min():.1%}")
    print(f"    Mean Window:     {rolling_hitrate.mean():.1%}")
    print(f"    % Above B/E:    {np.mean(rolling_hitrate > breakeven):.1%}")


# ---------------------------------------------------------------------------
# TEST 11: Edge Sensitivity Sweep
# ---------------------------------------------------------------------------
def test_edge_sensitivity(df):
    print_header("TEST 11: EDGE THRESHOLD SENSITIVITY")

    thresholds = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
    rows = []

    for t in thresholds:
        tdf = df[df["edge"] >= t]
        if len(tdf) == 0:
            continue
        n = len(tdf)
        wins = int(tdf["won"].sum())
        pnl = tdf["pnl"].sum()
        rows.append({
            "Min Edge": f"{t:.1%}",
            "Bets": n,
            "Hit%": f"{wins/n:.1%}",
            "PnL": f"{pnl:+.2f}",
            "ROI": f"{pnl/n*100:+.1f}%"
        })

    print(pd.DataFrame(rows).to_string(index=False))

    # Find optimal threshold
    best = max(rows, key=lambda r: float(r["PnL"].replace("+", "")))
    print(f"\n  Best threshold by PnL: {best['Min Edge']} ({best['PnL']} units, {best['Bets']} bets)")


# ---------------------------------------------------------------------------
# TEST 12: Recent Form Analysis
# ---------------------------------------------------------------------------
def test_recent_form(df):
    print_header("TEST 12: RECENT vs HISTORICAL PERFORMANCE")

    # Split into halves
    mid = len(df) // 2
    first_half = df.iloc[:mid]
    second_half = df.iloc[mid:]

    for label, subset in [("First Half (older)", first_half), ("Second Half (recent)", second_half)]:
        n = len(subset)
        wins = int(subset["won"].sum())
        pnl = subset["pnl"].sum()
        print(f"  {label}:")
        print(f"    Bets: {n}  |  Hit: {wins/n:.1%}  |  PnL: {pnl:+.2f}  |  ROI: {pnl/n*100:+.1f}%")

    # Last 3 seasons
    recent_seasons = sorted(df["season"].unique())[-3:]
    recent = df[df["season"].isin(recent_seasons)]
    n = len(recent)
    wins = int(recent["won"].sum())
    pnl = recent["pnl"].sum()
    print(f"\n  Last 3 Seasons ({recent_seasons[0]}-{recent_seasons[-1]}):")
    print(f"    Bets: {n}  |  Hit: {wins/n:.1%}  |  PnL: {pnl:+.2f}  |  ROI: {pnl/n*100:+.1f}%")

    # Last 1 season
    last_season = df["season"].max()
    last = df[df["season"] == last_season]
    n = len(last)
    wins = int(last["won"].sum())
    pnl = last["pnl"].sum()
    print(f"\n  Most Recent Season ({last_season}):")
    print(f"    Bets: {n}  |  Hit: {wins/n:.1%}  |  PnL: {pnl:+.2f}  |  ROI: {pnl/n*100:+.1f}%")


# ---------------------------------------------------------------------------
# FINAL VERDICT
# ---------------------------------------------------------------------------
def final_verdict(overall, significance, risk):
    print_header("FINAL VERDICT")

    checks = []

    # Profitability
    if overall["total_pnl"] > 0:
        checks.append(("Positive PnL", True))
    else:
        checks.append(("Positive PnL", False))

    # Statistical significance
    if significance["binom_p"] < 0.05:
        checks.append(("Binomial Sig (p<0.05)", True))
    elif significance["binom_p"] < 0.10:
        checks.append(("Binomial Sig (p<0.10)", True))
    else:
        checks.append(("Binomial Sig (p<0.10)", False))

    # Bootstrap
    ci_low, ci_high = significance["bootstrap_ci"]
    if ci_low > 0:
        checks.append(("Bootstrap CI > 0", True))
    else:
        checks.append(("Bootstrap CI > 0", False))

    # Monte Carlo
    if significance["mc_pvalue"] < 0.05:
        checks.append(("Monte Carlo Sig", True))
    else:
        checks.append(("Monte Carlo Sig", False))

    # Profit Factor
    if risk["profit_factor"] > 1.0:
        checks.append(("Profit Factor > 1", True))
    else:
        checks.append(("Profit Factor > 1", False))

    print("  Checklist:")
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {label}")

    passing = sum(1 for _, p in checks if p)
    total = len(checks)

    print(f"\n  Score: {passing}/{total}")

    if passing == total:
        verdict = "STRONG EDGE -- Model shows statistically significant profitability"
    elif passing >= 3:
        verdict = "MODERATE EVIDENCE -- Suggestive but not conclusive"
    elif passing >= 2:
        verdict = "WEAK EVIDENCE -- More data needed before betting real money"
    else:
        verdict = "NO EDGE -- Results consistent with random noise"

    print(f"\n  VERDICT: {verdict}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("  NFL SPREAD MODEL -- COMPREHENSIVE RESULTS TESTING")
    print("  Walk-Forward Backtest Evaluation")
    print("=" * 70)

    df = load_bets()
    print(f"\n  Loaded {len(df)} bets from {df['season'].min()} to {df['season'].max()}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    overall = test_overall_performance(df)
    significance = test_statistical_significance(df)
    risk = test_risk_metrics(df)
    test_season_breakdown(df)
    test_home_away_bias(df)
    test_edge_analysis(df)
    test_calibration(df)
    test_streak_analysis(df)
    test_drawdown_analysis(df)
    test_rolling_performance(df)
    test_edge_sensitivity(df)
    test_recent_form(df)
    final_verdict(overall, significance, risk)


if __name__ == "__main__":
    main()
