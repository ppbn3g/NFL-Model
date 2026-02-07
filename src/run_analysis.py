"""
Run full profitability analysis on the NFL spread model.

Usage: python src/run_analysis.py
"""
import os
import json
import numpy as np
import pandas as pd
from train import walk_forward_backtest
from backtest import edge_sensitivity_analysis
from stats_analysis import generate_profitability_report


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    print("=" * 60)
    print("NFL SPREAD MODEL -- PROFITABILITY ANALYSIS")
    print("=" * 60)

    # Step 1: Walk-forward backtest
    print("\n[1/4] Running walk-forward backtest...")
    all_bets, season_summaries, all_predicted = walk_forward_backtest(min_edge=0.01)

    if all_bets.empty:
        print("No bets generated. Check model and data.")
        return

    print(f"\nGenerated {len(all_bets)} bets across "
          f"{all_bets['season'].nunique()} seasons.")
    os.makedirs("data", exist_ok=True)
    all_bets.to_csv("data/walkforward_bets.csv", index=False)

    # Step 2: Statistical analysis
    print("\n[2/4] Running statistical profitability analysis...")
    report = generate_profitability_report(all_bets)

    # Step 3: Edge sensitivity
    print("\n[3/4] Running edge threshold sensitivity analysis...")
    # Use the original predicted data (with p_home), not the bet output
    sensitivity = edge_sensitivity_analysis(all_predicted)
    os.makedirs("reports", exist_ok=True)
    sensitivity.to_csv("reports/edge_sensitivity.csv", index=False)

    # Step 4: Print report
    print("\n[4/4] RESULTS")
    print("=" * 60)
    _print_report(report)

    # Save report
    _save_report(report)


def _print_report(report):
    b = report["basic"]
    print(f"\n--- BASIC STATS ---")
    print(f"  Total Bets:    {b['total_bets']}")
    print(f"  Win/Loss:      {b['total_wins']}/{b['total_losses']}")
    print(f"  Hit Rate:      {b['hit_rate']:.1%}")
    print(f"  Total PnL:     {b['total_pnl']:+.2f} units")
    print(f"  ROI per Bet:   {b['roi_per_bet']:+.2%}")

    bs = report["bootstrap"]
    print(f"\n--- BOOTSTRAP CONFIDENCE INTERVAL ({bs['ci_level']:.0%}) ---")
    print(f"  ROI: {bs['ci_lower']:+.2%} to {bs['ci_upper']:+.2%}")
    print(f"  P(ROI > 0):   p = {bs['p_value_positive']:.4f}")

    bt = report["binomial_test"]
    print(f"\n--- BINOMIAL TEST (vs {bt['breakeven_rate']:.2%} break-even) ---")
    print(f"  Observed Rate: {bt['observed_rate']:.2%}")
    print(f"  p-value:       {bt['p_value']:.4f}")
    print(f"  Significant at 5%: {'YES' if bt['is_significant_5pct'] else 'NO'}")

    r = report["risk"]
    print(f"\n--- RISK METRICS ---")
    print(f"  Sharpe Ratio:      {r['sharpe']:.3f}")
    print(f"  Sortino Ratio:     {r['sortino']:.3f}")
    print(f"  Max Drawdown:      {r['max_drawdown']:+.2f} units")
    print(f"  Max DD Duration:   {r['max_dd_duration']} bets")
    print(f"  Profit Factor:     {r['profit_factor']:.2f}")

    mc = report["monte_carlo"]
    print(f"\n--- MONTE CARLO (vs random betting) ---")
    print(f"  Null Mean PnL:     {mc['null_mean_pnl']:+.2f}")
    print(f"  Observed PnL:      {mc['observed_pnl']:+.2f}")
    print(f"  p-value:           {mc['p_value']:.4f}")

    if isinstance(report.get("by_season"), pd.DataFrame) and not report["by_season"].empty:
        print(f"\n--- SEASON BREAKDOWN ---")
        print(report["by_season"].to_string(index=False))

    clv = report.get("clv", {})
    if clv.get("available"):
        print(f"\n--- CLOSING LINE VALUE ---")
        print(f"  Mean CLV:          {clv['mean_clv']:+.4f}")
        print(f"  % Positive CLV:    {clv['pct_positive_clv']:.1%}")
    else:
        print(f"\n--- CLOSING LINE VALUE ---")
        print(f"  {clv.get('note', 'Not available')}")

    v = report["verdict"]
    print(f"\n{'=' * 60}")
    print(f"  VERDICT: {v['conclusion']}")
    print(f"  {v['recommendation']}")
    print(f"{'=' * 60}")


def _save_report(report):
    serializable = {}
    for key, val in report.items():
        if isinstance(val, pd.DataFrame):
            serializable[key] = val.to_dict(orient="records")
        elif isinstance(val, dict):
            clean = {}
            for k, v in val.items():
                if isinstance(v, (np.floating, float)):
                    clean[k] = float(v)
                elif isinstance(v, (np.integer, int)):
                    clean[k] = int(v)
                elif isinstance(v, np.ndarray):
                    continue  # skip large arrays like sim_pnls
                else:
                    clean[k] = v
            serializable[key] = clean
        else:
            serializable[key] = val

    os.makedirs("reports", exist_ok=True)
    with open("reports/profitability_report.json", "w") as f:
        json.dump(serializable, f, indent=2, cls=_NumpyEncoder)
    print("\nReport saved to reports/profitability_report.json")


if __name__ == "__main__":
    main()
