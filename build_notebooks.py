"""
build_notebooks.py
──────────────────
Generate all 5 Jupyter notebooks for the repository.
"""
import json, os

os.makedirs('notebooks', exist_ok=True)

def md(src):
    return {"cell_type":"markdown","metadata":{},"source": src if isinstance(src,list) else [src]}

def code(src, outputs=None):
    return {"cell_type":"code","execution_count":None,"metadata":{},
            "outputs": outputs or [],"source": src if isinstance(src,list) else [src]}

def nb(cells):
    return {"nbformat":4,"nbformat_minor":5,
            "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                        "language_info":{"name":"python","version":"3.10.0"}},
            "cells": cells}

SETUP = code([
    "import sys, os\n",
    "sys.path.insert(0, os.path.abspath('..'))\n",
    "from src.data_loader import load_data\n",
    "from src.plotting    import *\n",
    "from src.metrics     import compute_metrics\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy.stats import norm\n",
    "\n",
    "DATA_PATH = '../data/spy_simulated_2020_2022.csv'\n",
    "# Uncomment to use real data:\n",
    "# DATA_PATH = '../data/spy_2020_2022.csv'\n",
    "\n",
    "df = load_data(DATA_PATH)\n",
    "print('Data loaded:', df.shape)"
])

# ══════════════════════════════════════════════════════════
# NOTEBOOK 1 — BUY-WRITE
# ══════════════════════════════════════════════════════════
cells_bw = [
    md("# 📈 Notebook 1 — Buy-Write (Covered Call)\n> SPY Weekly ATM Covered Call  |  7 DTE  |  Δ≈0.50  |  Mid execution\n\n---"),
    md("## 1. Theory\n\nThe **buy-write** sells a call against a long equity position, capturing the **variance risk premium** (IV > realized vol) while reducing effective delta.\n\n$$\\text{P\\&L} = (S_T - S_0) + c - \\max(S_T - K, 0)$$\n\n**Greeks:** Δ≈+0.50, Γ<0, Θ>0, ν<0 — short volatility, long theta."),
    code([
        "# Payoff diagram\n",
        "fig, axes = plt.subplots(1,2,figsize=(14,5),facecolor=BG)\n",
        "S0,K,c = 100,100,3\n",
        "ST = np.linspace(70,135,400)\n",
        "pnl_spy = ST-S0; pnl_bw = pnl_spy+c-np.maximum(ST-K,0)\n",
        "for ax,title,y1,y2,col1,col2,lab1,lab2 in [\n",
        "    (axes[0],'P&L at Expiration',pnl_bw,pnl_spy,GOLD,BLUE,'Buy-Write','SPY'),\n",
        "    (axes[1],'Decomposition',pnl_bw,c-np.maximum(ST-K,0),GOLD,RED,'Combination','Short Call')]:\n",
        "    ax.plot(ST,y1,color=col1,lw=2.5,label=lab1)\n",
        "    ax.plot(ST,y2,color=col2,lw=1.5,ls='--',label=lab2,alpha=0.8)\n",
        "    ax.fill_between(ST,y1,y2,where=y1>=y2,alpha=0.12,color=col1)\n",
        "    ax.axvline(K,color=RED,lw=1,ls=':',alpha=0.7,label=f'K={K}')\n",
        "    style_ax(ax,title); legend(ax)\n",
        "    ax.set_xlabel('SPY at expiry'); ax.set_ylabel('P&L ($)')\n",
        "fig.suptitle('Buy-Write Payoff (S0=100, K=100, c=3)',color=WHITE,fontsize=12,fontweight='bold')\n",
        "fig.patch.set_facecolor(BG); plt.tight_layout(); plt.show()"
    ]),
    md("## 2. Strategy Construction\n\nAt each **roll date** (weekly expiration):\n1. Select call with DTE≈7 and Δ≈0.50\n2. Sell at mid price → `Cash += premium`\n3. At expiration: `Cash -= max(S_T - K, 0)`\n\n$$V_t = S_t + \\text{Cash}_t - \\text{Call}_{\\text{MTM},t}$$"),
    SETUP,
    code([
        "from src.strategy_buywrite import build_buywrite\n",
        "bw = build_buywrite(df, target_dte=7, target_delta=0.50)\n",
        "bw[['spy_price','index_level','spy_bnh','call_strike','call_premium','call_iv']].tail(10)"
    ]),
    md("## 3. Performance Dashboard"),
    code([
        "from src.plotting import plot_strategy_dashboard\n",
        "plot_strategy_dashboard(bw, 'Buy-Write', GOLD,\n",
        "                        save_path='../outputs/chart_buywrite.png')"
    ]),
    md("## 4. Metrics"),
    code([
        "m = compute_metrics(bw)\n",
        "for k,v in m.items():\n",
        "    if isinstance(v,float): print(f'  {k:<25} {v:.4f}')"
    ]),
]

# ══════════════════════════════════════════════════════════
# NOTEBOOK 2 — PROTECTIVE PUT
# ══════════════════════════════════════════════════════════
cells_pp = [
    md("# 🛡️ Notebook 2 — Protective Put\n> SPY Monthly OTM Put  |  30 DTE  |  Δ≈−0.30  |  Mid execution\n\n---"),
    md("## 1. Theory\n\nThe protective put is portfolio **insurance**: long SPY + long OTM put.\n\n$$\\text{P\\&L} = \\max(S_T, K_p) - S_0 - p$$\n\n- **Max loss**: $K_p - S_0 - p$ (bounded)\n- **Upside**: unlimited (minus premium cost)\n- **Equivalent**: long call + cash (put-call parity)\n\n**Greeks:** Δ≈+0.70, Γ>0, Θ<0, ν>0 — **long volatility**."),
    code([
        "# Payoff diagram\n",
        "fig, axes = plt.subplots(1,2,figsize=(14,5),facecolor=BG)\n",
        "S0,Kp,p = 100,97,2.5\n",
        "ST = np.linspace(75,130,500)\n",
        "pnl_spy = ST-S0\n",
        "pnl_put = np.maximum(Kp-ST,0)-p\n",
        "pnl_pp  = pnl_spy+pnl_put\n",
        "axes[0].plot(ST,pnl_spy,color=BLUE, lw=1.5,ls='--',label='SPY seul',alpha=0.7)\n",
        "axes[0].plot(ST,pnl_pp, color=GREEN,lw=2.5,label='Protective Put')\n",
        "axes[0].axvline(Kp,color=RED,lw=1.2,ls=':',label=f'K_put={Kp}')\n",
        "axes[0].fill_between(ST,pnl_pp,pnl_spy,where=pnl_pp>pnl_spy,alpha=0.15,color=GREEN)\n",
        "style_ax(axes[0],'P&L at Expiration'); legend(axes[0])\n",
        "axes[0].set_xlabel('SPY at expiry'); axes[0].set_ylabel('P&L ($)')\n",
        "axes[1].plot(ST,pnl_spy,color=BLUE,lw=1.5,ls='--',label='Long SPY',alpha=0.7)\n",
        "axes[1].plot(ST,pnl_put,color=RED, lw=1.5,ls='--',label=f'Long Put K={Kp}',alpha=0.8)\n",
        "axes[1].plot(ST,pnl_pp, color=GREEN,lw=2.5,label='Combined')\n",
        "style_ax(axes[1],'Decomposition'); legend(axes[1])\n",
        "axes[1].set_xlabel('SPY at expiry'); axes[1].set_ylabel('P&L ($)')\n",
        "fig.suptitle(f'Protective Put (S0={S0}, Kp={Kp}, p={p})',color=WHITE,fontsize=12,fontweight='bold')\n",
        "fig.patch.set_facecolor(BG); plt.tight_layout(); plt.show()"
    ]),
    md("## 2. Strategy Construction\n\nAt each **roll date** (monthly expiration):\n1. Select put with DTE≈30 and Δ≈−0.30\n2. Buy at mid → `Cash -= premium`\n3. At expiration: `Cash += max(K_p - S_T, 0)`\n\n$$V_t = S_t + \\text{Cash}_t + \\text{Put}_{\\text{MTM},t}$$"),
    SETUP,
    code([
        "from src.strategy_protective_put import build_protective_put\n",
        "pp = build_protective_put(df, target_dte=30, target_delta=-0.30)\n",
        "pp[['spy_price','index_level','spy_bnh','put_strike','put_premium','put_iv']].tail(10)"
    ]),
    md("## 3. Performance Dashboard"),
    code([
        "from src.plotting import plot_strategy_dashboard\n",
        "plot_strategy_dashboard(pp, 'Protective Put', GREEN,\n",
        "                        save_path='../outputs/chart_protective_put.png')"
    ]),
    md("## 4. Metrics"),
    code([
        "m = compute_metrics(pp)\n",
        "for k,v in m.items():\n",
        "    if isinstance(v,float): print(f'  {k:<25} {v:.4f}')"
    ]),
]

# ══════════════════════════════════════════════════════════
# NOTEBOOK 3 — COLLAR
# ══════════════════════════════════════════════════════════
cells_col = [
    md("# 🔒 Notebook 3 — Collar\n> Monthly Zero-Cost Collar  |  30 DTE  |  Put Δ≈−0.25  |  Call Δ≈+0.25\n\n---"),
    md("## 1. Theory\n\nThe collar combines a **protective put** (hedging) and a **short call** (financing):\n\n$$\\text{P\\&L} = \\begin{cases} K_p - S_0 - (p-c) & S_T < K_p \\\\ S_T - S_0 - (p-c) & K_p \\leq S_T \\leq K_c \\\\ K_c - S_0 - (p-c) & S_T > K_c \\end{cases}$$\n\n**Zero-cost collar**: select $K_c$ such that $c \\approx p$ → no net premium.\n\n**Greeks:** reduced Δ, short Γ, near-zero ν (long put ≈ short call in vega)."),
    code([
        "fig, axes = plt.subplots(1,2,figsize=(14,5),facecolor=BG)\n",
        "S0,Kp,p,Kc,c = 100,95,2.0,105,2.0\n",
        "ST = np.linspace(75,130,500)\n",
        "pnl_spy = ST-S0\n",
        "pnl_collar = pnl_spy+np.maximum(Kp-ST,0)-p+(c-np.maximum(ST-Kc,0))\n",
        "axes[0].plot(ST,pnl_spy,   color=BLUE,  lw=1.5,ls='--',label='SPY',alpha=0.7)\n",
        "axes[0].plot(ST,pnl_collar,color=PURPLE,lw=2.5,label='Collar')\n",
        "axes[0].axvline(Kp,color=RED, lw=1.2,ls=':',label=f'Put K={Kp}')\n",
        "axes[0].axvline(Kc,color=GOLD,lw=1.2,ls=':',label=f'Call K={Kc}')\n",
        "style_ax(axes[0],'P&L at Expiration'); legend(axes[0])\n",
        "axes[0].set_xlabel('SPY at expiry'); axes[0].set_ylabel('P&L ($)')\n",
        "axes[1].plot(ST,pnl_spy,color=BLUE,lw=1.5,ls='--',label='SPY',alpha=0.7)\n",
        "axes[1].plot(ST,np.maximum(Kp-ST,0)-p,color=RED, lw=1.5,ls='--',label=f'Long Put {Kp}',alpha=0.8)\n",
        "axes[1].plot(ST,c-np.maximum(ST-Kc,0),color=GOLD,lw=1.5,ls='--',label=f'Short Call {Kc}',alpha=0.8)\n",
        "axes[1].plot(ST,pnl_collar,color=PURPLE,lw=2.5,label='Collar')\n",
        "style_ax(axes[1],'Decomposition'); legend(axes[1])\n",
        "axes[1].set_xlabel('SPY at expiry'); axes[1].set_ylabel('P&L ($)')\n",
        "fig.suptitle(f'Zero-Cost Collar (Kp={Kp}, Kc={Kc})',color=WHITE,fontsize=12,fontweight='bold')\n",
        "fig.patch.set_facecolor(BG); plt.tight_layout(); plt.show()"
    ]),
    md("## 2. Strategy Construction\n\n$$V_t = S_t + \\text{Cash}_t + \\text{Put}_{\\text{MTM},t} - \\text{Call}_{\\text{MTM},t}$$\n\n**Cash at roll**: `Cash += C_mid - P_mid` (net, often ≈ 0)"),
    SETUP,
    code([
        "from src.strategy_collar import build_collar\n",
        "col = build_collar(df, target_dte=30, put_delta=-0.25, call_delta=0.25)\n",
        "col[['spy_price','index_level','spy_bnh','put_strike','call_strike','net_cost']].tail(10)"
    ]),
    md("## 3. Performance Dashboard"),
    code([
        "from src.plotting import plot_strategy_dashboard\n",
        "plot_strategy_dashboard(col, 'Collar', PURPLE,\n",
        "                        save_path='../outputs/chart_collar.png')"
    ]),
    md("## 4. Metrics"),
    code([
        "m = compute_metrics(col)\n",
        "for k,v in m.items():\n",
        "    if isinstance(v,float): print(f'  {k:<25} {v:.4f}')"
    ]),
]

# ══════════════════════════════════════════════════════════
# NOTEBOOK 4 — LONG STRADDLE
# ══════════════════════════════════════════════════════════
cells_std = [
    md("# ⚡ Notebook 4 — Long Straddle\n> Monthly ATM Straddle  |  30 DTE  |  Δ≈±0.50  |  Long Volatility\n\n---"),
    md("## 1. Theory\n\nThe long straddle is a **pure volatility** bet:\n\n$$\\text{P\\&L} = |S_T - K| - (c + p)$$\n\n**Break-evens**: $K \\pm (c+p)$\n\nProfitable when **realized vol > implied vol**. The exact opposite of the Buy-Write.\n\n**Greeks:** Δ≈0, Γ>>0, Θ<<0, ν>>0 — maximum long vega."),
    code([
        "fig, axes = plt.subplots(1,2,figsize=(14,5),facecolor=BG)\n",
        "K,c,p = 100,3.2,3.0; tot = c+p\n",
        "ST = np.linspace(75,130,500)\n",
        "pnl = np.abs(ST-K)-tot\n",
        "axes[0].plot(ST,pnl,color=GOLD,lw=2.5,label='Long Straddle')\n",
        "axes[0].fill_between(ST,pnl,0,where=pnl>0,alpha=0.2,color=GOLD)\n",
        "axes[0].fill_between(ST,pnl,0,where=pnl<0,alpha=0.15,color=RED)\n",
        "axes[0].axvline(K-tot,color=RED,  lw=1,ls=':',label=f'BE down={K-tot}')\n",
        "axes[0].axvline(K+tot,color=GREEN,lw=1,ls=':',label=f'BE up={K+tot}')\n",
        "axes[0].axvline(K,    color=WHITE, lw=1,ls='--',alpha=0.6,label=f'ATM K={K}')\n",
        "style_ax(axes[0],'P&L at Expiration'); legend(axes[0])\n",
        "axes[0].set_xlabel('SPY at expiry'); axes[0].set_ylabel('P&L ($)')\n",
        "axes[1].plot(ST,np.maximum(ST-K,0)-c,color=BLUE,lw=1.5,ls='--',label=f'Long Call K={K}',alpha=0.8)\n",
        "axes[1].plot(ST,np.maximum(K-ST,0)-p,color=RED, lw=1.5,ls='--',label=f'Long Put K={K}',alpha=0.8)\n",
        "axes[1].plot(ST,pnl,color=GOLD,lw=2.5,label='Straddle')\n",
        "style_ax(axes[1],'Decomposition'); legend(axes[1])\n",
        "axes[1].set_xlabel('SPY at expiry'); axes[1].set_ylabel('P&L ($)')\n",
        "fig.suptitle(f'Long Straddle (K={K}, total prem={tot})',color=WHITE,fontsize=12,fontweight='bold')\n",
        "fig.patch.set_facecolor(BG); plt.tight_layout(); plt.show()"
    ]),
    md("## 2. Strategy Construction\n\n**No underlying position.**\n\n$$V_t = \\text{Cash}_t + \\text{Call}_{\\text{MTM},t} + \\text{Put}_{\\text{MTM},t}$$\n\nAt roll: `Cash -= (C_mid + P_mid)`  \nAt expiry: `Cash += payoff_call + payoff_put`"),
    SETUP,
    code([
        "from src.strategy_straddle import build_straddle\n",
        "std = build_straddle(df, target_dte=30)\n",
        "std[['spy_price','index_level','spy_bnh','strike','total_premium','call_iv']].tail(10)"
    ]),
    md("## 3. Performance Dashboard"),
    code([
        "from src.plotting import plot_strategy_dashboard\n",
        "plot_strategy_dashboard(std, 'Long Straddle', RED,\n",
        "                        save_path='../outputs/chart_straddle.png')"
    ]),
    md("## 4. Metrics"),
    code([
        "m = compute_metrics(std)\n",
        "for k,v in m.items():\n",
        "    if isinstance(v,float): print(f'  {k:<25} {v:.4f}')"
    ]),
]

# ══════════════════════════════════════════════════════════
# NOTEBOOK 5 — COMPARISON
# ══════════════════════════════════════════════════════════
cells_cmp = [
    md("# 📊 Notebook 5 — Full Comparison\n> All Four Strategies vs SPY Buy & Hold  |  2020–2022\n\n---"),
    md("## 1. Overview\n\n| Strategy | Bias | Cost | Vega | Theta |\n|---|---|---|---|---|\n| Buy-Write | Long, capped | Credit | Short | Long |\n| Protective Put | Long, protected | Debit | Long | Short |\n| Collar | Neutral-Long | ~Zero | ~Neutral | ~Neutral |\n| Long Straddle | Volatility | Debit | Long | Short |"),
    SETUP,
    code([
        "from src.strategy_buywrite       import build_buywrite\n",
        "from src.strategy_protective_put  import build_protective_put\n",
        "from src.strategy_collar          import build_collar\n",
        "from src.strategy_straddle        import build_straddle\n",
        "from src.metrics                  import metrics_table\n",
        "from src.plotting                 import plot_comparison_dashboard\n",
        "\n",
        "bw  = build_buywrite(df)\n",
        "pp  = build_protective_put(df)\n",
        "col = build_collar(df)\n",
        "std = build_straddle(df)\n",
        "\n",
        "spy_ref = bw[['spy_bnh','spy_price']].copy()\n",
        "results_dict = {\n",
        "    'Buy-Write'      : bw,\n",
        "    'Protective Put' : pp,\n",
        "    'Collar'         : col,\n",
        "    'Long Straddle'  : std,\n",
        "    'SPY B&H'        : spy_ref,\n",
        "}\n",
        "print('All strategies built.')"
    ]),
    md("## 2. Comparison Dashboard"),
    code([
        "plot_comparison_dashboard(results_dict,\n",
        "                          save_path='../outputs/chart_comparison.png')"
    ]),
    md("## 3. Metrics Table"),
    code([
        "mt = metrics_table(results_dict)\n",
        "mt"
    ]),
    md("## 4. Correlation of Daily Returns"),
    code([
        "rets = pd.DataFrame({\n",
        "    n: (res['spy_bnh' if n=='SPY B&H' else 'index_level']).pct_change()\n",
        "    for n, res in results_dict.items()\n",
        "})\n",
        "corr = rets.corr()\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(8,6), facecolor=BG)\n",
        "ax.set_facecolor(PANEL)\n",
        "im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)\n",
        "plt.colorbar(im, ax=ax)\n",
        "ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=30, color=WHITE, fontsize=9)\n",
        "ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index, color=WHITE, fontsize=9)\n",
        "for i in range(len(corr)):\n",
        "    for j in range(len(corr)):\n",
        "        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', color='black', fontsize=8, fontweight='bold')\n",
        "for s in ax.spines.values(): s.set_edgecolor(GREY)\n",
        "ax.set_title('Correlation of Daily Returns', color=WHITE, fontsize=11, fontweight='bold')\n",
        "fig.patch.set_facecolor(BG)\n",
        "plt.tight_layout(); plt.show()"
    ]),
]

# ── Write notebooks ──────────────────────────────────────────────────
notebooks = {
    'notebooks/01_buywrite.ipynb'        : cells_bw,
    'notebooks/02_protective_put.ipynb'  : cells_pp,
    'notebooks/03_collar.ipynb'          : cells_col,
    'notebooks/04_straddle.ipynb'        : cells_std,
    'notebooks/05_comparison.ipynb'      : cells_cmp,
}

for path, cells in notebooks.items():
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb(cells), f, indent=1, default=str, ensure_ascii=False)
    print(f"✅ {path}  ({len(cells)} cells)")
