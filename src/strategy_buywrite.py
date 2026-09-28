"""
strategy_buywrite.py
────────────────────
Weekly ATM Buy-Write (Covered Call) index on SPY.

Rules
-----
- Long 1 unit SPY (normalised)
- Short 1 ATM call  ≈ 7 DTE, delta ≈ 0.50
- Execution at mid  (C_BID + C_ASK) / 2
- Roll at expiration (or when DTE == 0)

P&L
---
V_t = SPY_t + Cash_t - Call_MTM_t

Cash updates:
  +premium  at roll (received)
  -intrinsic at expiration if ITM
"""

import numpy as np
import pandas as pd
from src.data_loader import select_option, get_trading_days, get_day


# ── Parameters ────────────────────────────────────────────────────────
DEFAULT_DTE   = 7
DEFAULT_DELTA = 0.50   # ATM call


def build_buywrite(df: pd.DataFrame,
                   target_dte: int   = DEFAULT_DTE,
                   target_delta: float = DEFAULT_DELTA) -> pd.DataFrame:
    """
    Build the Buy-Write index.

    Parameters
    ----------
    df           : clean options DataFrame from data_loader.load_data()
    target_dte   : desired DTE at roll (default 7)
    target_delta : desired call delta (default 0.50)

    Returns
    -------
    pd.DataFrame indexed by date with columns:
        spy_price, portfolio_value, index_level, spy_bnh,
        cash, call_mtm, call_strike, call_expiry, call_premium,
        call_delta, call_iv, call_dte_sold
    """
    trading_days = get_trading_days(df)
    records      = []

    cash         = 0.0
    current_call = None
    spy_entry    = None
    base_port    = None

    for date in trading_days:
        df_today  = get_day(df, date)
        spy_price = df_today['UNDERLYING_LAST'].iloc[0]

        if spy_entry is None:
            spy_entry = spy_price

        # ── 1. Settle expired call ────────────────────────────────
        if current_call is not None and current_call['expiry'] == date:
            intrinsic    = max(spy_price - current_call['strike'], 0.0)
            cash        -= intrinsic
            current_call = None

        # ── 2. Roll: sell new call ────────────────────────────────
        if current_call is None:
            row = select_option(df_today, 'call', target_dte, target_delta)
            if row is not None:
                premium  = float(row['C_MID'])
                cash    += premium
                current_call = {
                    'strike'  : float(row['STRIKE']),
                    'expiry'  : row['EXPIRE_DATE'],
                    'premium' : premium,
                    'delta'   : float(row['C_DELTA']),
                    'iv'      : float(row['C_IV']) if pd.notna(row.get('C_IV')) else np.nan,
                    'dte_sold': float(row['DTE']),
                }

        # ── 3. Mark-to-market call ────────────────────────────────
        call_mtm = 0.0
        if current_call is not None:
            mask  = ((df_today['STRIKE']      == current_call['strike']) &
                     (df_today['EXPIRE_DATE'] == current_call['expiry']))
            match = df_today[mask]
            if not match.empty:
                call_mtm = float(match.iloc[0]['C_MID'])

        # ── 4. Portfolio valuation ────────────────────────────────
        pv = spy_price + cash - call_mtm
        if base_port is None:
            base_port = pv

        records.append({
            'date'           : date,
            'spy_price'      : spy_price,
            'portfolio_value': pv,
            'index_level'    : 100.0 * pv / base_port,
            'spy_bnh'        : 100.0 * spy_price / spy_entry,
            'cash'           : cash,
            'call_mtm'       : call_mtm,
            'call_strike'    : current_call['strike']   if current_call else np.nan,
            'call_expiry'    : current_call['expiry']   if current_call else pd.NaT,
            'call_premium'   : current_call['premium']  if current_call else np.nan,
            'call_delta'     : current_call['delta']    if current_call else np.nan,
            'call_iv'        : current_call['iv']       if current_call else np.nan,
            'call_dte_sold'  : current_call['dte_sold'] if current_call else np.nan,
        })

    result = pd.DataFrame(records).set_index('date')
    _print_summary(result, 'Buy-Write')
    return result


def _print_summary(res: pd.DataFrame, name: str) -> None:
    n_rolls = res['call_premium'].notna().diff().fillna(0).lt(0).sum()
    print(f"✅  {name} index built")
    print(f"    Days      : {len(res)}")
    print(f"    Rolls     : {res.drop_duplicates(subset=['call_expiry']).shape[0]}")
    print(f"    Avg IV    : {res['call_iv'].mean():.2%}")
    print(f"    Avg prem  : {res['call_premium'].mean():.3f}")
