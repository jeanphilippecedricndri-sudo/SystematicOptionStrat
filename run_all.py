"""
run_all.py
──────────
Run all four strategies and produce outputs.

Usage
-----
    python run_all.py [path/to/spy_options.csv]
"""

import sys
import os
import pandas as pd

# ── ensure src/ is on path ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader          import load_data
from src.strategy_buywrite    import build_buywrite
from src.strategy_protective_put import build_protective_put
from src.strategy_collar      import build_collar
from src.strategy_straddle    import build_straddle
from src.metrics              import metrics_table
from src.plotting             import plot_strategy_dashboard, plot_comparison_dashboard
from src.plotting             import GOLD, GREEN, PURPLE, RED, BLUE


def main(filepath: str):
    os.makedirs('outputs', exist_ok=True)

    print("\n" + "="*60)
    print("  SPY OPTIONS STRATEGIES — FULL BACKTEST")
    print("="*60 + "\n")

    # ── Load data ─────────────────────────────────────────────────
    df = load_data(filepath)

    # ── Build indices ─────────────────────────────────────────────
    print("\n── Building strategies ──────────────────────────────────")
    bw  = build_buywrite(df,       target_dte=7,  target_delta=0.50)
    pp  = build_protective_put(df, target_dte=30, target_delta=-0.30)
    col = build_collar(df,         target_dte=30, put_delta=-0.25, call_delta=0.25)
    std = build_straddle(df,       target_dte=30)

    # ── Export CSVs ───────────────────────────────────────────────
    print("\n── Exporting CSVs ───────────────────────────────────────")
    bw.to_csv('outputs/buywrite_index.csv');       print("  outputs/buywrite_index.csv")
    pp.to_csv('outputs/protective_put_index.csv'); print("  outputs/protective_put_index.csv")
    col.to_csv('outputs/collar_index.csv');        print("  outputs/collar_index.csv")
    std.to_csv('outputs/straddle_index.csv');      print("  outputs/straddle_index.csv")

    # ── Individual dashboards ─────────────────────────────────────
    print("\n── Generating dashboards ────────────────────────────────")
    plot_strategy_dashboard(bw,  'Buy-Write',      GOLD,   'outputs/chart_buywrite.png')
    plot_strategy_dashboard(pp,  'Protective Put', GREEN,  'outputs/chart_protective_put.png')
    plot_strategy_dashboard(col, 'Collar',         PURPLE, 'outputs/chart_collar.png')
    plot_strategy_dashboard(std, 'Long Straddle',  RED,    'outputs/chart_straddle.png')

    # ── Comparison dashboard ──────────────────────────────────────
    spy_ref = bw[['spy_bnh', 'spy_price']].copy()
    results_dict = {
        'Buy-Write'      : bw,
        'Protective Put' : pp,
        'Collar'         : col,
        'Long Straddle'  : std,
        'SPY B&H'        : spy_ref,
    }
    plot_comparison_dashboard(results_dict, save_path='outputs/chart_comparison.png')

    # ── Metrics table ─────────────────────────────────────────────
    print("\n── Performance Summary ──────────────────────────────────")
    mt = metrics_table(results_dict)
    print(mt.to_string())
    mt.to_csv('outputs/metrics_summary.csv')
    print("\n  outputs/metrics_summary.csv")

    print("\n" + "="*60)
    print("  DONE — all outputs in outputs/")
    print("="*60 + "\n")


if __name__ == '__main__':
    fp = sys.argv[1] if len(sys.argv) > 1 else 'data/spy_2020_2022.csv'
    main(fp)
