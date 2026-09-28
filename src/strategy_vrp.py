"""
strategy_vrp.py
───────────────
Variance Risk Premium (VRP) strategies on SPY.

Two implementations:

1. SHORT STRADDLE
   - Sell ATM call + put each week (~7 DTE, delta ≈ ±0.50)
   - Roll at expiration
   - Pure short-vol: profits when IV > RV

2. SYNTHETIC VARIANCE SWAP (log-contract replication)
   - Replicate a variance swap via a portfolio of OTM options
     weighted by 1/K² (Demeterfi et al. 1999 / CBOE VIX methodology)
   - Fair variance strike: K_var = (2/T) * Σ ΔK_i/K_i² * Option_i
   - At expiration: P&L = Notional * (RV² - IV²)
   - Roll monthly (~30 DTE)

Both strategies also compute:
   - Realized volatility (close-to-close 21d)
   - Realized volatility (Parkinson high-low estimator)
   - VRP signal = IV - RV (annualised)
"""

import numpy as np
import pandas as pd
from src.data_loader import get_trading_days, get_day, select_option


# ══════════════════════════════════════════════════════════════════════
# REALIZED VOLATILITY ESTIMATORS
# ══════════════════════════════════════════════════════════════════════

def realized_vol_cc(spy_series: pd.Series, window: int = 21) -> pd.Series:
    """
    Close-to-close realized volatility (annualised).
    rv_t = std(log returns over window) * sqrt(252)
    """
    log_ret = np.log(spy_series / spy_series.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def realized_vol_parkinson(df_daily: pd.DataFrame,
                            window: int = 21) -> pd.Series:
    """
    Parkinson (1980) high-low range estimator (annualised).
    More efficient than close-to-close (uses intraday range).

    Estimator:
        RV_park = sqrt( 1/(4*ln2) * E[(ln H/L)²] * 252 )

    Since we only have close prices in the options data, we approximate
    H = Close * (1 + |daily_return|/2) and L = Close * (1 - |daily_return|/2).
    For real data with OHLC this can be replaced with actual H/L.

    Parameters
    ----------
    df_daily : DataFrame with 'spy_price' indexed by date
    window   : rolling window in days
    """
    S    = df_daily['spy_price']
    ret  = np.log(S / S.shift(1)).abs()
    # Approximate HL range from daily return magnitude
    hl2  = (ret ** 2) / (4 * np.log(2))
    park = np.sqrt(hl2.rolling(window).mean() * 252)
    return park


def compute_vrp_signal(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Compute daily VRP signal from ATM implied vol vs realized vol.
    Returns DataFrame with: spy_price, iv_atm, rv_cc, rv_park, vrp_cc, vrp_park
    """
    # ATM IV per day: nearest-to-money call with DTE closest to 21
    def get_atm_iv(grp):
        grp = grp.dropna(subset=['C_IV','C_DELTA'])
        grp = grp[grp['C_BID'] > 0]
        if grp.empty:
            return np.nan
        # use DTE closest to 21
        grp2 = grp.copy()
        grp2['_dd'] = (grp2['DTE'] - 21).abs()
        min_dte = grp2['_dd'].min()
        sub = grp2[grp2['_dd'] == min_dte]
        # ATM = delta closest to 0.50
        sub = sub.copy()
        sub['_ad'] = (sub['C_DELTA'] - 0.50).abs()
        return float(sub.loc[sub['_ad'].idxmin(), 'C_IV'])

    atm_iv = df.groupby('QUOTE_DATE').apply(get_atm_iv).rename('iv_atm')

    # Daily SPY prices
    spy_daily = (
        df.groupby('QUOTE_DATE')['UNDERLYING_LAST'].first()
        .rename('spy_price')
    )

    daily = pd.DataFrame({'spy_price': spy_daily, 'iv_atm': atm_iv})
    daily['rv_cc']    = realized_vol_cc(daily['spy_price'], window)
    daily['rv_park']  = realized_vol_parkinson(daily, window)
    daily['vrp_cc']   = daily['iv_atm'] - daily['rv_cc']
    daily['vrp_park'] = daily['iv_atm'] - daily['rv_park']
    return daily.dropna()


# ══════════════════════════════════════════════════════════════════════
# STRATEGY 1: SHORT STRADDLE
# ══════════════════════════════════════════════════════════════════════

def build_short_straddle(df: pd.DataFrame,
                          target_dte: int = 7) -> pd.DataFrame:
    """
    Weekly short straddle index.

    At each roll:
      - Sell ATM call (Δ≈+0.50) + ATM put (Δ≈−0.50), same strike/expiry
      - Cash += call_mid + put_mid

    At expiration:
      - Cash -= max(S_T - K, 0)   [call settled]
      - Cash -= max(K - S_T, 0)   [put settled]

    Portfolio value (no underlying):
      V_t = Cash_t - Call_MTM_t - Put_MTM_t
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

        # ── 1. Settle at expiration ────────────────────────────
        if current_call is not None and current_call['expiry'] <= date:
            cash -= max(spy_price - current_call['strike'], 0.0)  # short call loss
            cash -= max(current_put['strike'] - spy_price,  0.0)  # short put loss
            current_call = current_put = None

        # ── 2. Roll: sell new straddle ─────────────────────────
        if current_call is None:
            call_row = select_option(df_today, 'call', target_dte, 0.50)
            if call_row is not None:
                atm_strike = float(call_row['STRIKE'])
                atm_expiry = call_row['EXPIRE_DATE']

                # put at same strike
                mask = ((df_today['EXPIRE_DATE'] == atm_expiry) &
                        (df_today['P_BID'] > 0))
                put_cands = df_today[mask].copy()
                if not put_cands.empty:
                    put_cands['_sd'] = (put_cands['STRIKE'] - atm_strike).abs()
                    put_row = put_cands.loc[put_cands['_sd'].idxmin()]

                    call_prem = float(call_row['C_MID'])
                    put_prem  = float(put_row['P_MID'])
                    cash     += call_prem + put_prem   # receive both premiums

                    current_call = {'strike': atm_strike,  'expiry': atm_expiry,
                                    'premium': call_prem,
                                    'iv': float(call_row.get('C_IV', np.nan))}
                    current_put  = {'strike': float(put_row['STRIKE']),
                                    'expiry': atm_expiry,
                                    'premium': put_prem,
                                    'iv': float(put_row.get('P_IV', np.nan))}

        # ── 3. Mark-to-market ──────────────────────────────────
        call_mtm = put_mtm = 0.0
        if current_call is not None:
            m = df_today[(df_today['STRIKE']      == current_call['strike']) &
                         (df_today['EXPIRE_DATE'] == current_call['expiry'])]
            if not m.empty: call_mtm = float(m.iloc[0]['C_MID'])
        if current_put is not None:
            m = df_today[(df_today['STRIKE']      == current_put['strike']) &
                         (df_today['EXPIRE_DATE'] == current_put['expiry'])]
            if not m.empty: put_mtm = float(m.iloc[0]['P_MID'])

        # ── 4. Portfolio: short straddle = cash - calls - puts ─
        total_prem = ((current_call['premium'] + current_put['premium'])
                      if current_call else np.nan)
        pv = cash - call_mtm - put_mtm
        if base_port is None:
            # Anchor: first total premium received
            base_port = float(total_prem) if (total_prem and not np.isnan(total_prem) and total_prem > 0) else 10.0

        # Index: starts at 100, moves proportionally to P&L vs base
        index_level = 100.0 + (pv / base_port) * 100.0

        records.append({
            'date'          : date,
            'spy_price'     : spy_price,
            'portfolio_value': pv,
            'index_level'   : index_level,
            'spy_bnh'       : 100.0 * spy_price / spy_entry,
            'cash'          : cash,
            'call_mtm'      : call_mtm,
            'put_mtm'       : put_mtm,
            'straddle_mtm'  : call_mtm + put_mtm,
            'strike'        : current_call['strike']  if current_call else np.nan,
            'total_premium' : total_prem,
            'call_iv'       : current_call['iv']      if current_call else np.nan,
        })

    result = pd.DataFrame(records).set_index('date')
    print(f"✅  Short Straddle (VRP) index built")
    print(f"    Days      : {len(result)}")
    print(f"    Rolls     : {result.drop_duplicates(subset=['strike']).shape[0]}")
    print(f"    Avg prem  : {result['total_premium'].mean():.3f}")
    return result


# ══════════════════════════════════════════════════════════════════════
# STRATEGY 2: SYNTHETIC VARIANCE SWAP
# ══════════════════════════════════════════════════════════════════════

def compute_fair_variance_strike(df_day: pd.DataFrame,
                                  target_dte: int = 30,
                                  r: float = 0.04) -> dict:
    """
    Compute the fair variance strike (annualised) via the
    Demeterfi-Derman-Kamal-Zou (1999) log-contract replication:

        K_var = (2/T) * [ Σ_{puts}  (ΔK/K²) * P(K)
                        + Σ_{calls} (ΔK/K²) * C(K) ]

    where the sum is over OTM options for a given expiration.
    """
    if df_day.empty:
        return None

    # Find expiry closest to target_dte
    df = df_day.copy()
    df['_dte_diff'] = (df['DTE'] - target_dte).abs()
    best_dte = df['_dte_diff'].min()
    chain    = df[df['_dte_diff'] == best_dte].copy().sort_values('STRIKE')

    if len(chain) < 5:
        return None

    T   = float(chain['DTE'].iloc[0]) / 365.0
    if T <= 0:
        return None
    S0  = float(chain['UNDERLYING_LAST'].iloc[0])
    F   = S0 * np.exp(r * T)

    # ATM IV (for reference)
    atm_idx = (chain['STRIKE'] - F).abs().values.argmin()
    iv_atm  = float(chain.iloc[atm_idx]['C_IV'])
    K_atm   = float(chain.iloc[atm_idx]['STRIKE'])

    # OTM puts (K < K_atm) and OTM calls (K >= K_atm)
    puts  = chain[chain['STRIKE'] <  K_atm].copy()
    calls = chain[chain['STRIKE'] >= K_atm].copy()

    if puts.empty or calls.empty:
        return None

    def integrate_leg(rows, price_col):
        K_arr = rows['STRIKE'].values.astype(float)
        P_arr = rows[price_col].values.astype(float)
        dk    = np.gradient(K_arr)
        return float(np.sum(dk / (K_arr ** 2) * P_arr))

    contrib_puts  = integrate_leg(puts,  'P_MID')
    contrib_calls = integrate_leg(calls, 'C_MID')

    K_var = (2.0 / T) * (contrib_puts + contrib_calls)
    K_var = max(K_var, 1e-8)
    K_vol = np.sqrt(K_var)

    return {
        'K_var'   : K_var,
        'K_vol'   : K_vol,
        'S0'      : S0,
        'F'       : F,
        'T'       : T,
        'dte'     : float(chain['DTE'].iloc[0]),
        'n_puts'  : len(puts),
        'n_calls' : len(calls),
        'iv_atm'  : iv_atm,
    }


def build_variance_swap(df: pd.DataFrame,
                         target_dte: int  = 30,
                         notional_vega: float = 1000.0,
                         r: float = 0.04) -> pd.DataFrame:
    """
    Synthetic Variance Swap index.

    At each roll (monthly):
      - Compute K_var (fair variance strike) from OTM option chain
      - Enter long variance position: notional_vega dollars per vol point
      - Vega notional N_vega → variance notional N_var = N_vega / (2 * K_vol)

    At expiration:
      - Realized variance = annualised variance of daily log-returns over period
      - P&L = N_var * (RV² - K_var)  [long variance = profit if RV > IV]

    For the SHORT VARIANCE strategy (VRP harvest): flip sign.

    Daily MTM (approximate):
      P&L_t ≈ N_var * (RV²_running - K_var) * (days_elapsed / T_total)

    Parameters
    ----------
    notional_vega : $ P&L per 1 vol point move (e.g. 1000 = $1000 per 1% vol move)
    """
    trading_days = get_trading_days(df)
    records      = []

    cash      = 0.0
    current   = None    # current variance swap leg
    spy_entry = None
    base_port = None

    # Pre-compute daily spy prices
    spy_series = pd.Series(
        {d: get_day(df, d)['UNDERLYING_LAST'].iloc[0] for d in trading_days}
    )

    for i, date in enumerate(trading_days):
        df_today  = get_day(df, date)
        spy_price = spy_series[date]
        if spy_entry is None:
            spy_entry = spy_price

        # ── 1. Settle expired var swap ─────────────────────────
        if current is not None and date >= current['expiry']:
            # Compute realized variance over the holding period
            mask     = (spy_series.index >= current['start_date']) & \
                       (spy_series.index <= date)
            s_window = spy_series[mask]
            if len(s_window) > 2:
                log_rets = np.log(s_window / s_window.shift(1)).dropna()
                rv2      = log_rets.var() * 252   # annualised variance
            else:
                rv2 = current['K_var']

            # SHORT variance: we sold variance at K_var, buy back at RV²
            # P&L = N_var * (K_var - RV²) for short variance
            pnl      = current['N_var'] * (current['K_var'] - rv2)
            cash    += pnl
            current  = None

        # ── 2. Roll: open new var swap ─────────────────────────
        if current is None:
            vs = compute_fair_variance_strike(df_today, target_dte, r)
            if vs is not None and vs['K_var'] > 0:
                # Short variance: sell at K_var (receive variance premium)
                N_var = notional_vega / (2.0 * vs['K_vol'])

                # Find expiry date
                df_exp = df_today[
                    (df_today['DTE'] - target_dte).abs() ==
                    (df_today['DTE'] - target_dte).abs().min()
                ]
                expiry_date = df_exp['EXPIRE_DATE'].iloc[0]

                current = {
                    'K_var'     : vs['K_var'],
                    'K_vol'     : vs['K_vol'],
                    'N_var'     : N_var,
                    'iv_atm'    : vs['iv_atm'],
                    'T'         : vs['T'],
                    'dte'       : vs['dte'],
                    'expiry'    : expiry_date,
                    'start_date': date,
                }

        # ── 3. Daily MTM (mark-to-forward) ────────────────────
        mtm_pnl = 0.0
        rv2_running = np.nan
        if current is not None:
            mask     = (spy_series.index >= current['start_date']) & \
                       (spy_series.index <= date)
            s_window = spy_series[mask]
            if len(s_window) > 2:
                log_rets    = np.log(s_window / s_window.shift(1)).dropna()
                rv2_running = log_rets.var() * 252
                days_elapsed = len(log_rets)
                days_total   = max(current['dte'], 1)
                # Interpolated MTM
                weight   = days_elapsed / days_total
                mtm_pnl  = current['N_var'] * (
                    weight * (current['K_var'] - rv2_running)
                )

        # ── 4. Portfolio value ─────────────────────────────────
        pv = cash + mtm_pnl
        if base_port is None:
            base_port = notional_vega   # anchor to vega notional

        records.append({
            'date'           : date,
            'spy_price'      : spy_price,
            'portfolio_value': pv,
            'index_level'    : 100.0 + pv / base_port * 100,
            'spy_bnh'        : 100.0 * spy_price / spy_entry,
            'cash'           : cash,
            'mtm_pnl'        : mtm_pnl,
            'K_var'          : current['K_var']   if current else np.nan,
            'K_vol'          : current['K_vol']   if current else np.nan,
            'rv2_running'    : rv2_running,
            'rv_running'     : np.sqrt(rv2_running) if not np.isnan(rv2_running) else np.nan,
            'iv_atm'         : current['iv_atm']  if current else np.nan,
            'vrp_live'       : ((current['K_vol'] - np.sqrt(rv2_running))
                                if (current and not np.isnan(rv2_running)) else np.nan),
        })

    result = pd.DataFrame(records).set_index('date')
    print(f"✅  Variance Swap (short variance) index built")
    print(f"    Days             : {len(result)}")
    print(f"    Avg K_vol (IV)   : {result['K_vol'].mean():.2%}")
    print(f"    Avg RV (running) : {result['rv_running'].mean():.2%}")
    print(f"    Avg VRP live     : {result['vrp_live'].mean():.2%}")
    return result
