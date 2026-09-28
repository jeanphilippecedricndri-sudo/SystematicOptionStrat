"""
strategy_collar.py
──────────────────
Monthly Collar index on SPY (near zero-cost).

Rules
-----
- Long 1 unit SPY (normalised)
- Long 1 OTM put   ≈ 30 DTE, delta ≈ -0.25
- Short 1 OTM call ≈ 30 DTE, delta ≈ +0.25  (same expiration)
- Execution at mid for both legs
- Roll at expiration

P&L
---
V_t = SPY_t + Cash_t + Put_MTM_t - Call_MTM_t

Cash updates at roll:
  -premium_put  (paid)
  +premium_call (received)
"""

import numpy as np
import pandas as pd
from src.data_loader import select_option, get_trading_days, get_day


DEFAULT_DTE        = 30
DEFAULT_PUT_DELTA  = -0.25
DEFAULT_CALL_DELTA =  0.25


def build_collar(df: pd.DataFrame,
                 target_dte:  int   = DEFAULT_DTE,
                 put_delta:   float = DEFAULT_PUT_DELTA,
                 call_delta:  float = DEFAULT_CALL_DELTA) -> pd.DataFrame:
    """
    Build the Collar index.

    Parameters
    ----------
    df           : clean options DataFrame
    target_dte   : desired DTE at roll (default 30)
    put_delta    : desired put delta  (default -0.25)
    call_delta   : desired call delta (default +0.25)

    Returns
    -------
    pd.DataFrame indexed by date.
    """
    trading_days = get_trading_days(df)
    records      = []

    cash         = 0.0
    current_put  = None
    current_call = None
    spy_entry    = None
    base_port    = None

    for date in trading_days:
        df_today  = get_day(df, date)
        spy_price = df_today['UNDERLYING_LAST'].iloc[0]

        if spy_entry is None:
            spy_entry = spy_price

        # ── 1. Settle expired legs ───────────────────────────────
        if current_put is not None and current_put['expiry'] <= date:
            cash += max(current_put['strike']  - spy_price, 0.0)  # put  ITM
            cash -= max(spy_price - current_call['strike'], 0.0)  # call ITM
            current_put = current_call = None

        # ── 2. Roll: open both legs ──────────────────────────────
        if current_put is None:
            put_row  = select_option(df_today, 'put',  target_dte, put_delta)
            call_row = select_option(df_today, 'call', target_dte, call_delta)

            if put_row is not None and call_row is not None:
                put_prem  = float(put_row['P_MID'])
                call_prem = float(call_row['C_MID'])
                cash     -= put_prem    # pay put
                cash     += call_prem   # receive call

                current_put = {
                    'strike'  : float(put_row['STRIKE']),
                    'expiry'  : put_row['EXPIRE_DATE'],
                    'premium' : put_prem,
                    'delta'   : float(put_row['P_DELTA']),
                    'iv'      : float(put_row['P_IV']) if pd.notna(put_row.get('P_IV')) else np.nan,
                }
                current_call = {
                    'strike'  : float(call_row['STRIKE']),
                    'expiry'  : call_row['EXPIRE_DATE'],
                    'premium' : call_prem,
                    'delta'   : float(call_row['C_DELTA']),
                    'iv'      : float(call_row['C_IV']) if pd.notna(call_row.get('C_IV')) else np.nan,
                }

        # ── 3. Mark-to-market both legs ──────────────────────────
        put_mtm = call_mtm = 0.0
        if current_put is not None:
            m = df_today[(df_today['STRIKE']      == current_put['strike']) &
                         (df_today['EXPIRE_DATE'] == current_put['expiry'])]
            if not m.empty:
                put_mtm = float(m.iloc[0]['P_MID'])

        if current_call is not None:
            m = df_today[(df_today['STRIKE']      == current_call['strike']) &
                         (df_today['EXPIRE_DATE'] == current_call['expiry'])]
            if not m.empty:
                call_mtm = float(m.iloc[0]['C_MID'])

        # ── 4. Portfolio valuation ────────────────────────────────
        pv = spy_price + cash + put_mtm - call_mtm
        if base_port is None:
            base_port = pv

        net_cost = ((current_put['premium']  if current_put  else 0) -
                    (current_call['premium'] if current_call else 0))

        records.append({
            'date'           : date,
            'spy_price'      : spy_price,
            'portfolio_value': pv,
            'index_level'    : 100.0 * pv / base_port,
            'spy_bnh'        : 100.0 * spy_price / spy_entry,
            'cash'           : cash,
            'put_mtm'        : put_mtm,
            'call_mtm'       : call_mtm,
            'put_strike'     : current_put['strike']   if current_put  else np.nan,
            'call_strike'    : current_call['strike']  if current_call else np.nan,
            'put_premium'    : current_put['premium']  if current_put  else np.nan,
            'call_premium'   : current_call['premium'] if current_call else np.nan,
            'net_cost'       : net_cost,
            'put_iv'         : current_put['iv']       if current_put  else np.nan,
            'call_iv'        : current_call['iv']      if current_call else np.nan,
        })

    result = pd.DataFrame(records).set_index('date')
    _print_summary(result)
    return result


def _print_summary(res: pd.DataFrame) -> None:
    print(f"✅  Collar index built")
    print(f"    Days      : {len(res)}")
    print(f"    Rolls     : {res.drop_duplicates(subset=['put_strike','call_strike']).shape[0]}")
    print(f"    Avg put IV: {res['put_iv'].mean():.2%}")
    print(f"    Avg net cost per roll: {res['net_cost'].mean():.3f}")
