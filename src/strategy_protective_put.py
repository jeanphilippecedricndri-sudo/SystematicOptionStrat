"""
strategy_protective_put.py
──────────────────────────
Monthly Protective Put index on SPY.

Rules
-----
- Long 1 unit SPY (normalised)
- Long 1 OTM put  ≈ 30 DTE, delta ≈ -0.30
- Execution at mid  (P_BID + P_ASK) / 2
- Roll at expiration

P&L
---
V_t = SPY_t + Cash_t + Put_MTM_t

Cash updates:
  -premium  at roll (paid)
  +intrinsic at expiration if ITM
"""

import numpy as np
import pandas as pd
from src.data_loader import select_option, get_trading_days, get_day


DEFAULT_DTE   = 30
DEFAULT_DELTA = -0.30   # OTM put


def build_protective_put(df: pd.DataFrame,
                         target_dte: int     = DEFAULT_DTE,
                         target_delta: float = DEFAULT_DELTA) -> pd.DataFrame:
    """
    Build the Protective Put index.

    Parameters
    ----------
    df           : clean options DataFrame
    target_dte   : desired DTE at roll (default 30)
    target_delta : desired put delta (default -0.30)

    Returns
    -------
    pd.DataFrame indexed by date.
    """
    trading_days = get_trading_days(df)
    records      = []

    cash        = 0.0
    current_put = None
    spy_entry   = None
    base_port   = None

    for date in trading_days:
        df_today  = get_day(df, date)
        spy_price = df_today['UNDERLYING_LAST'].iloc[0]

        if spy_entry is None:
            spy_entry = spy_price

        # ── 1. Settle expired put ────────────────────────────────
        if current_put is not None and current_put['expiry'] <= date:
            intrinsic   = max(current_put['strike'] - spy_price, 0.0)
            cash       += intrinsic   # receive intrinsic
            current_put = None

        # ── 2. Roll: buy new put ─────────────────────────────────
        if current_put is None:
            row = select_option(df_today, 'put', target_dte, target_delta)
            if row is not None:
                premium  = float(row['P_MID'])
                cash    -= premium   # pay premium
                current_put = {
                    'strike'  : float(row['STRIKE']),
                    'expiry'  : row['EXPIRE_DATE'],
                    'premium' : premium,
                    'delta'   : float(row['P_DELTA']),
                    'iv'      : float(row['P_IV']) if pd.notna(row.get('P_IV')) else np.nan,
                    'dte_sold': float(row['DTE']),
                }

        # ── 3. Mark-to-market put ────────────────────────────────
        put_mtm = 0.0
        if current_put is not None:
            mask  = ((df_today['STRIKE']      == current_put['strike']) &
                     (df_today['EXPIRE_DATE'] == current_put['expiry']))
            match = df_today[mask]
            if not match.empty:
                put_mtm = float(match.iloc[0]['P_MID'])

        # ── 4. Portfolio valuation ────────────────────────────────
        pv = spy_price + cash + put_mtm
        if base_port is None:
            base_port = pv

        records.append({
            'date'           : date,
            'spy_price'      : spy_price,
            'portfolio_value': pv,
            'index_level'    : 100.0 * pv / base_port,
            'spy_bnh'        : 100.0 * spy_price / spy_entry,
            'cash'           : cash,
            'put_mtm'        : put_mtm,
            'put_strike'     : current_put['strike']   if current_put else np.nan,
            'put_expiry'     : current_put['expiry']   if current_put else pd.NaT,
            'put_premium'    : current_put['premium']  if current_put else np.nan,
            'put_delta'      : current_put['delta']    if current_put else np.nan,
            'put_iv'         : current_put['iv']       if current_put else np.nan,
            'put_dte_sold'   : current_put['dte_sold'] if current_put else np.nan,
        })

    result = pd.DataFrame(records).set_index('date')
    _print_summary(result, 'Protective Put')
    return result


def _print_summary(res: pd.DataFrame, name: str) -> None:
    print(f"✅  {name} index built")
    print(f"    Days      : {len(res)}")
    print(f"    Rolls     : {res.drop_duplicates(subset=['put_expiry']).shape[0]}")
    print(f"    Avg IV    : {res['put_iv'].mean():.2%}")
    print(f"    Avg prem  : {res['put_premium'].mean():.3f}")
