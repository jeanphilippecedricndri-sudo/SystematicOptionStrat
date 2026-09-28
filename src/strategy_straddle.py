"""
strategy_straddle.py
────────────────────
Monthly Long Straddle index on SPY.

Rules
-----
- No position on the underlying
- Long 1 ATM call ≈ 30 DTE, delta ≈ +0.50
- Long 1 ATM put  ≈ 30 DTE, same strike as call
- Execution at mid
- Roll at expiration

P&L
---
V_t = Cash_t + Call_MTM_t + Put_MTM_t

Cash updates at roll:
  -(premium_call + premium_put)  paid
At expiration:
  +max(S_T - K, 0)  call payoff
  +max(K - S_T, 0)  put payoff
"""

import numpy as np
import pandas as pd
from src.data_loader import select_option, get_trading_days, get_day


DEFAULT_DTE = 30


def build_straddle(df: pd.DataFrame,
                   target_dte: int = DEFAULT_DTE) -> pd.DataFrame:
    """
    Build the Long Straddle index.

    Parameters
    ----------
    df         : clean options DataFrame
    target_dte : desired DTE at roll (default 30)

    Returns
    -------
    pd.DataFrame indexed by date.
    """
    trading_days = get_trading_days(df)
    records      = []

    cash         = 0.0
    current_call = None
    current_put  = None
    spy_entry    = None
    base_port    = None

    for date in trading_days:
        df_today  = get_day(df, date)
        spy_price = df_today['UNDERLYING_LAST'].iloc[0]

        if spy_entry is None:
            spy_entry = spy_price

        # ── 1. Settle expired legs ───────────────────────────────
        if current_call is not None and current_call['expiry'] <= date:
            cash += max(spy_price - current_call['strike'], 0.0)  # call payoff
            cash += max(current_put['strike'] - spy_price,  0.0)  # put payoff
            current_call = current_put = None

        # ── 2. Roll: buy call + put ──────────────────────────────
        if current_call is None:
            call_row = select_option(df_today, 'call', target_dte, 0.50)

            if call_row is not None:
                atm_strike = float(call_row['STRIKE'])
                atm_expiry = call_row['EXPIRE_DATE']

                # Put: same strike (or nearest)
                mask_put = ((df_today['EXPIRE_DATE'] == atm_expiry) &
                            (df_today['P_BID'] > 0))
                put_cands = df_today[mask_put].copy()

                if not put_cands.empty:
                    put_cands['_sd'] = (put_cands['STRIKE'] - atm_strike).abs()
                    put_row = put_cands.loc[put_cands['_sd'].idxmin()]

                    call_prem = float(call_row['C_MID'])
                    put_prem  = float(put_row['P_MID'])
                    total_prem = call_prem + put_prem
                    cash      -= total_prem

                    current_call = {
                        'strike'  : atm_strike,
                        'expiry'  : atm_expiry,
                        'premium' : call_prem,
                        'iv'      : float(call_row['C_IV']) if pd.notna(call_row.get('C_IV')) else np.nan,
                        'dte_sold': float(call_row['DTE']),
                    }
                    current_put = {
                        'strike'  : float(put_row['STRIKE']),
                        'expiry'  : atm_expiry,
                        'premium' : put_prem,
                        'iv'      : float(put_row['P_IV']) if pd.notna(put_row.get('P_IV')) else np.nan,
                    }

        # ── 3. Mark-to-market both legs ──────────────────────────
        call_mtm = put_mtm = 0.0
        if current_call is not None:
            m = df_today[(df_today['STRIKE']      == current_call['strike']) &
                         (df_today['EXPIRE_DATE'] == current_call['expiry'])]
            if not m.empty:
                call_mtm = float(m.iloc[0]['C_MID'])

        if current_put is not None:
            m = df_today[(df_today['STRIKE']      == current_put['strike']) &
                         (df_today['EXPIRE_DATE'] == current_put['expiry'])]
            if not m.empty:
                put_mtm = float(m.iloc[0]['P_MID'])

        # ── 4. Portfolio valuation (no SPY) ──────────────────────
        pv = cash + call_mtm + put_mtm
        if base_port is None:
            # Anchor to first total premium paid so index starts at 100
            total_first = ((current_call['premium'] + current_put['premium'])
                           if current_call else 1.0)
            base_port = total_first

        total_prem = ((current_call['premium'] + current_put['premium'])
                      if current_call else np.nan)

        records.append({
            'date'           : date,
            'spy_price'      : spy_price,
            'portfolio_value': pv,
            'index_level'    : 100.0 + (pv / base_port) * 100,
            'spy_bnh'        : 100.0 * spy_price / spy_entry,
            'cash'           : cash,
            'call_mtm'       : call_mtm,
            'put_mtm'        : put_mtm,
            'straddle_mtm'   : call_mtm + put_mtm,
            'strike'         : current_call['strike']  if current_call else np.nan,
            'expiry'         : current_call['expiry']  if current_call else pd.NaT,
            'total_premium'  : total_prem,
            'call_iv'        : current_call['iv']      if current_call else np.nan,
            'put_iv'         : current_put['iv']       if current_put  else np.nan,
        })

    result = pd.DataFrame(records).set_index('date')
    _print_summary(result)
    return result


def _print_summary(res: pd.DataFrame) -> None:
    print(f"✅  Long Straddle index built")
    print(f"    Days            : {len(res)}")
    print(f"    Rolls           : {res.drop_duplicates(subset=['strike','expiry']).shape[0]}")
    print(f"    Avg total prem  : {res['total_premium'].mean():.3f}")
    print(f"    Avg call IV     : {res['call_iv'].mean():.2%}")
