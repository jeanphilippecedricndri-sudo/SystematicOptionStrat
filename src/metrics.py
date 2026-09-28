"""
metrics.py
──────────
Performance metrics for options strategy indices.
"""

import numpy as np
import pandas as pd


def compute_metrics(results: pd.DataFrame,
                    index_col: str = 'index_level',
                    rf: float = 0.0,
                    periods: int = 252) -> dict:
    """
    Compute a full set of performance metrics.

    Parameters
    ----------
    results   : DataFrame with at least `index_col`
    index_col : column name for the strategy index (base 100)
    rf        : annual risk-free rate
    periods   : trading days per year

    Returns
    -------
    dict of labelled metrics
    """
    idx = results[index_col].dropna()
    r   = idx.pct_change().dropna()
    n   = len(r)

    # ── Returns ──────────────────────────────────────────────────────
    total_ret  = idx.iloc[-1] / 100 - 1
    ann_ret    = (1 + r).prod() ** (periods / n) - 1

    # ── Risk ─────────────────────────────────────────────────────────
    ann_vol    = r.std() * np.sqrt(periods)
    downside   = r[r < 0].std() * np.sqrt(periods)
    skewness   = float(r.skew())
    kurtosis   = float(r.kurt())

    # ── Drawdown ─────────────────────────────────────────────────────
    running_max  = idx.cummax()
    drawdown_ser = (idx - running_max) / running_max
    max_dd       = float(drawdown_ser.min())
    # Average drawdown duration
    in_dd        = drawdown_ser < 0
    dd_dur       = in_dd.astype(int).groupby((~in_dd).cumsum()).sum()
    avg_dd_dur   = float(dd_dur[dd_dur > 0].mean()) if (dd_dur > 0).any() else 0.0

    # ── Risk-adjusted ─────────────────────────────────────────────────
    excess_r  = r - rf / periods
    sharpe    = excess_r.mean() / r.std() * np.sqrt(periods) if r.std() > 0 else np.nan
    sortino   = ann_ret / downside if downside > 0 else np.nan
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    omega_num = r[r > 0].sum()
    omega_den = abs(r[r < 0].sum())
    omega     = omega_num / omega_den if omega_den > 0 else np.nan

    # ── VaR / CVaR ────────────────────────────────────────────────────
    var_95  = float(np.percentile(r, 5))
    cvar_95 = float(r[r <= var_95].mean())

    return {
        # Returns
        'Total Return'      : total_ret,
        'Ann. Return'       : ann_ret,
        # Risk
        'Ann. Volatility'   : ann_vol,
        'Skewness'          : skewness,
        'Kurtosis'          : kurtosis,
        # Drawdown
        'Max Drawdown'      : max_dd,
        'Avg DD Duration'   : avg_dd_dur,
        # Risk-adjusted
        'Sharpe Ratio'      : sharpe,
        'Sortino Ratio'     : sortino,
        'Calmar Ratio'      : calmar,
        'Omega Ratio'       : omega,
        # Tail risk
        'VaR 95%'           : var_95,
        'CVaR 95%'          : cvar_95,
    }


def metrics_table(results_dict: dict,
                  index_col: str = 'index_level') -> pd.DataFrame:
    """
    Build a comparison DataFrame from multiple strategy results.

    Parameters
    ----------
    results_dict : {'Strategy Name': results_df, ...}
    index_col    : column to use as index level

    Returns
    -------
    pd.DataFrame  rows = strategies, columns = metrics
    """
    rows = {}
    for name, res in results_dict.items():
        col = 'spy_bnh' if name == 'SPY B&H' else index_col
        tmp = res.rename(columns={col: 'index_level'})
        rows[name] = compute_metrics(tmp)

    df = pd.DataFrame(rows).T

    # Format for display
    pct_cols  = ['Total Return','Ann. Return','Ann. Volatility',
                 'Max Drawdown','VaR 95%','CVaR 95%']
    float_cols = ['Sharpe Ratio','Sortino Ratio','Calmar Ratio',
                  'Omega Ratio','Skewness','Kurtosis','Avg DD Duration']

    fmt = df.copy().astype(object)
    for c in pct_cols:
        if c in fmt.columns:
            fmt[c] = df[c].apply(lambda x: f"{x:.2%}" if pd.notna(x) else 'N/A')
    for c in float_cols:
        if c in fmt.columns:
            fmt[c] = df[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else 'N/A')

    return fmt
