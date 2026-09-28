"""
generate_simulated_data.py
──────────────────────────
Generate realistic simulated SPY options data (2020–2022)
for use in notebooks when real data is not available.
Uses Black-Scholes to price options consistently.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# ── Black-Scholes pricing ──────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, opt_type='call'):
    if T <= 0:
        return max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta(S, K, T, r, sigma, opt_type='call'):
    if T <= 0:
        return (1.0 if S > K else 0.0) if opt_type == 'call' else (-1.0 if S < K else 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if opt_type == 'call' else norm.cdf(d1) - 1

def bs_iv_approx(S, K, T):
    """Approximate IV: ATM ≈ base_vol, skew for OTM puts"""
    moneyness = K / S
    base_vol  = 0.20
    skew      = -0.15 * (moneyness - 1)   # put skew
    vol_smile = 0.05 * (moneyness - 1)**2  # smile
    return max(0.08, base_vol + skew + vol_smile)


def generate_spy_path():
    """Simulate SPY price path 2020-01-02 → 2022-12-30 with realistic regimes."""
    start_date = date(2020, 1, 2)
    end_date   = date(2022, 12, 30)

    all_dates  = pd.bdate_range(start=start_date, end=end_date)
    n          = len(all_dates)

    # Regime parameters: (mu_ann, sigma_ann, length_approx)
    regimes = [
        (0.30, 0.15, 40),    # Jan 2020: bull
        (-6.0, 1.50, 25),    # Covid crash: Feb-Mar 2020
        (2.50, 0.45, 60),    # Recovery: Apr-Jun 2020
        (0.20, 0.20, 100),   # Summer 2020
        (0.25, 0.18, 100),   # Fall 2020
        (0.30, 0.15, 130),   # 2021 bull H1
        (0.15, 0.12, 120),   # 2021 H2
        (-0.25, 0.28, 130),  # 2022 bear
    ]

    prices    = [337.0]
    daily_vol = [0.20 / np.sqrt(252)]

    regime_idx    = 0
    regime_count  = 0
    regime_lengths = [r[2] for r in regimes]

    for i in range(1, n):
        mu_ann, sig_ann, _ = regimes[min(regime_idx, len(regimes)-1)]
        mu_d  = mu_ann  / 252
        sig_d = sig_ann / np.sqrt(252)

        ret = np.random.normal(mu_d, sig_d)
        prices.append(prices[-1] * np.exp(ret))
        daily_vol.append(sig_d)

        regime_count += 1
        if regime_idx < len(regimes)-1 and regime_count >= regime_lengths[regime_idx]:
            regime_idx  += 1
            regime_count = 0

    return pd.Series(prices, index=all_dates)


def generate_options_data(spy_path: pd.Series,
                          strikes_around: int = 20,
                          strike_step: float  = 1.0,
                          expirations_ahead: int = 5) -> pd.DataFrame:
    """
    For each trading day, generate option chain around SPY price.
    Expirations: next N Fridays (weekly + monthly).
    """
    r   = 0.04
    rows = []

    print("Generating simulated options chain...")

    for dt, spy in spy_path.items():
        quote_date = dt.date()

        # Find next N Fridays
        expiries = []
        d = dt.date() + timedelta(days=1)
        while len(expiries) < expirations_ahead:
            if d.weekday() == 4:   # Friday
                expiries.append(d)
            d += timedelta(days=1)

        # Strike grid centered on ATM
        atm_strike  = round(spy / strike_step) * strike_step
        strike_low  = atm_strike - strikes_around * strike_step
        strike_high = atm_strike + strikes_around * strike_step
        strikes     = np.arange(strike_low, strike_high + strike_step, strike_step)

        for exp in expiries:
            dte = (exp - quote_date).days
            T   = dte / 365.0
            if T <= 0:
                continue

            for K in strikes:
                sigma = bs_iv_approx(spy, K, T)
                sigma_p = bs_iv_approx(spy, K, T) * 1.02   # slight put skew

                c_price  = bs_price(spy, K, T, r, sigma,   'call')
                p_price  = bs_price(spy, K, T, r, sigma_p, 'put')
                c_delta  = bs_delta(spy, K, T, r, sigma,   'call')
                p_delta  = bs_delta(spy, K, T, r, sigma_p, 'put')

                # Bid/ask spread (tighter ATM)
                moneyness  = abs(K - spy) / spy
                spread_pct = 0.02 + 0.04 * moneyness
                c_spread   = max(0.01, c_price * spread_pct)
                p_spread   = max(0.01, p_price * spread_pct)

                rows.append({
                    'QUOTE_DATE'       : str(quote_date),
                    'QUOTE_TIME_HOURS' : 16.0,
                    'UNDERLYING_LAST'  : round(spy, 4),
                    'EXPIRE_DATE'      : str(exp),
                    'DTE'              : dte,
                    'STRIKE'           : K,
                    'C_DELTA'          : round(c_delta, 4),
                    'C_BID'            : round(max(0.01, c_price - c_spread/2), 4),
                    'C_ASK'            : round(max(0.02, c_price + c_spread/2), 4),
                    'C_IV'             : round(sigma, 4),
                    'C_LAST'           : round(c_price, 4),
                    'P_DELTA'          : round(p_delta, 4),
                    'P_BID'            : round(max(0.01, p_price - p_spread/2), 4),
                    'P_ASK'            : round(max(0.02, p_price + p_spread/2), 4),
                    'P_IV'             : round(sigma_p, 4),
                    'P_LAST'           : round(p_price, 4),
                    'STRIKE_DISTANCE'  : round(abs(K - spy), 4),
                    'STRIKE_DISTANCE_PCT': round(abs(K - spy) / spy, 4),
                })

        if (spy_path.index.get_loc(dt) + 1) % 100 == 0:
            print(f"  {quote_date} (SPY={spy:.2f})")

    df = pd.DataFrame(rows)
    print(f"Generated {len(df):,} rows over {spy_path.shape[0]} trading days")
    return df


if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok=True)

    spy = generate_spy_path()
    df  = generate_options_data(spy, strikes_around=15, expirations_ahead=4)

    out = 'data/spy_simulated_2020_2022.csv'
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}  ({len(df):,} rows)")
