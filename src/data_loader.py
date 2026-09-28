"""
data_loader.py
──────────────
Load and clean SPY options data.
Handles the ' [COL_NAME]' bracket notation automatically.
"""

import pandas as pd
import numpy as np


# ── Column name mapping (bracket → clean) ─────────────────────────────
REQUIRED_COLS = [
    'QUOTE_DATE', 'QUOTE_TIME_HOURS', 'UNDERLYING_LAST', 'EXPIRE_DATE', 'DTE',
    'C_DELTA', 'C_BID', 'C_ASK', 'C_IV',
    'P_DELTA', 'P_BID', 'P_ASK', 'P_IV',
    'STRIKE',
]


def load_data(filepath: str, eod_hour: float = 16.0) -> pd.DataFrame:
    """
    Load and clean SPY options CSV.

    Parameters
    ----------
    filepath : str
        Path to the CSV file. Columns may be formatted as ' [COL_NAME]'.
    eod_hour : float
        Hour to filter for end-of-day quotes (default 16.0).

    Returns
    -------
    pd.DataFrame with clean column names, parsed dates, numeric fields,
    C_MID and P_MID computed, illiquid options removed.
    """
    df = pd.read_csv(filepath, sep=',', skipinitialspace=True, low_memory=False)

    # ── Strip bracket notation: ' [QUOTE_DATE]' → 'QUOTE_DATE' ──────
    df.columns = (
        df.columns
        .str.strip()
        .str.strip('[]')
        .str.strip()
    )

    # ── Parse dates ───────────────────────────────────────────────────
    df['QUOTE_DATE']  = pd.to_datetime(df['QUOTE_DATE'].astype(str).str.strip())
    df['EXPIRE_DATE'] = pd.to_datetime(df['EXPIRE_DATE'].astype(str).str.strip())

    # ── Numeric conversion ────────────────────────────────────────────
    numeric_cols = [
        'UNDERLYING_LAST', 'DTE', 'STRIKE',
        'C_DELTA', 'C_BID', 'C_ASK', 'C_IV', 'C_VOLUME',
        'P_DELTA', 'P_BID', 'P_ASK', 'P_IV', 'P_VOLUME',
        'QUOTE_TIME_HOURS',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── Filter end-of-day quotes ──────────────────────────────────────
    if 'QUOTE_TIME_HOURS' in df.columns:
        df = df[df['QUOTE_TIME_HOURS'] == eod_hour]

    # ── Compute mid prices ────────────────────────────────────────────
    df['C_MID'] = (df['C_BID'] + df['C_ASK']) / 2
    df['P_MID'] = (df['P_BID'] + df['P_ASK']) / 2

    # ── Drop rows missing essentials ──────────────────────────────────
    df = df.dropna(subset=['UNDERLYING_LAST', 'STRIKE', 'DTE', 'EXPIRE_DATE'])
    df = df[df['DTE'] >= 0]

    df = df.sort_values('QUOTE_DATE').reset_index(drop=True)

    _print_summary(df, filepath)
    return df


def _print_summary(df: pd.DataFrame, filepath: str) -> None:
    print(f"✅  Loaded: {filepath}")
    print(f"    Rows            : {len(df):,}")
    print(f"    Period          : {df['QUOTE_DATE'].min().date()} → {df['QUOTE_DATE'].max().date()}")
    print(f"    Trading days    : {df['QUOTE_DATE'].nunique()}")
    print(f"    Unique strikes  : {df['STRIKE'].nunique()}")
    print(f"    Unique expiries : {df['EXPIRE_DATE'].nunique()}")


def get_trading_days(df: pd.DataFrame) -> list:
    """Return sorted list of unique trading dates."""
    return sorted(df['QUOTE_DATE'].unique())


def get_day(df: pd.DataFrame, date) -> pd.DataFrame:
    """Return all rows for a given date."""
    return df[df['QUOTE_DATE'] == date]


# ── Option selection helpers ──────────────────────────────────────────

def select_option(df_day: pd.DataFrame,
                  side: str,
                  target_dte: int,
                  target_delta: float) -> pd.Series:
    """
    Generic option selector.

    Parameters
    ----------
    df_day       : DataFrame for one trading day
    side         : 'call' or 'put'
    target_dte   : desired DTE
    target_delta : desired delta (e.g. 0.50 for call ATM, -0.30 for put OTM)

    Returns
    -------
    Best matching row or None.
    """
    if df_day.empty:
        return None

    prefix   = 'C' if side == 'call' else 'P'
    bid_col  = f'{prefix}_BID'
    mid_col  = f'{prefix}_MID'
    dlt_col  = f'{prefix}_DELTA'

    df = df_day.dropna(subset=[bid_col, dlt_col]).copy()
    df = df[df[bid_col] > 0]
    if df.empty:
        return None

    # Step 1: closest DTE
    df['_dte_diff'] = (df['DTE'] - target_dte).abs()
    best_dte        = df['_dte_diff'].min()
    cands           = df[df['_dte_diff'] == best_dte].copy()

    # Step 2: closest delta
    cands['_dlt_diff'] = (cands[dlt_col] - target_delta).abs()
    best_row           = cands.loc[cands['_dlt_diff'].idxmin()]
    return best_row
