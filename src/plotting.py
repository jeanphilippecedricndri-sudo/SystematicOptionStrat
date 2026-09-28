"""
plotting.py
───────────
Visualization utilities shared across all strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

# ── Palette ───────────────────────────────────────────────────────────
GOLD   = '#c9a84c'
BLUE   = '#4c9ac9'
RED    = '#c94c4c'
GREEN  = '#4cc98a'
PURPLE = '#9a4cc9'
GREY   = '#444444'
WHITE  = '#e8e8e8'
PANEL  = '#161616'
BG     = '#0d0d0d'

STRATEGY_COLORS = {
    'Buy-Write'      : GOLD,
    'Protective Put' : GREEN,
    'Collar'         : PURPLE,
    'Long Straddle'  : RED,
    'SPY B&H'        : BLUE,
}

PCT_FMT  = FuncFormatter(lambda x, _: f"{x:.1f}%")
DPCT_FMT = FuncFormatter(lambda x, _: f"{x*100:.1f}%")


def style_ax(ax, title: str = '', fontsize: int = 10):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.set_tick_params(rotation=20)
    for s in ax.spines.values():
        s.set_edgecolor(GREY)
    if title:
        ax.set_title(title, color=WHITE, fontsize=fontsize, fontweight='bold', pad=8)
    ax.yaxis.label.set_color(WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.grid(axis='y', color=GREY, lw=0.4, alpha=0.5)
    ax.axhline(0, color=GREY, lw=0.6)


def legend(ax, **kwargs):
    kwargs.setdefault('fontsize', 8)
    ax.legend(facecolor=PANEL, edgecolor=GREY, labelcolor=WHITE, **kwargs)


# ── Single-strategy dashboard ─────────────────────────────────────────

def plot_strategy_dashboard(results: pd.DataFrame,
                             strategy_name: str,
                             color: str = GOLD,
                             save_path: str = None):
    """
    5-panel dashboard for a single strategy vs SPY B&H.
    Panels: index level, drawdown, rolling vol, IV/premium at roll, metrics table.
    """
    from src.metrics import compute_metrics

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32)

    # ── Panel 1: Index ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(results.index, results['index_level'], color=color, lw=2,   label=strategy_name)
    ax1.plot(results.index, results['spy_bnh'],     color=BLUE,  lw=1.4, label='SPY B&H', alpha=0.8, ls='--')
    ax1.fill_between(results.index, results['index_level'], results['spy_bnh'],
                     where=results['index_level'] >= results['spy_bnh'], alpha=0.10, color=color)
    ax1.fill_between(results.index, results['index_level'], results['spy_bnh'],
                     where=results['index_level'] <  results['spy_bnh'], alpha=0.08, color=BLUE)
    style_ax(ax1, f'{strategy_name} vs SPY Buy & Hold  (Base = 100)', fontsize=11)
    legend(ax1, fontsize=9)
    # Limites Y dynamiques — marge de 5% autour des vraies valeurs
    all_vals = pd.concat([results['index_level'], results['spy_bnh']]).dropna()
    ymin = all_vals.min()
    ymax = all_vals.max()
    margin = (ymax - ymin) * 0.05
    ax1.set_ylim(ymin - margin, ymax + margin)

    # ── Panel 2: Drawdown ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    strat_dd = (results['index_level'] / results['index_level'].cummax() - 1) * 100
    spy_dd   = (results['spy_bnh']     / results['spy_bnh'].cummax()     - 1) * 100
    ax2.fill_between(results.index, strat_dd, 0, color=color, alpha=0.35, label=strategy_name)
    ax2.fill_between(results.index, spy_dd,   0, color=BLUE,  alpha=0.18, label='SPY')
    ax2.plot(results.index, strat_dd, color=color, lw=0.9)
    ax2.plot(results.index, spy_dd,   color=BLUE,  lw=0.9)
    style_ax(ax2, 'Drawdown (%)')
    legend(ax2)

    # ── Panel 3: Rolling volatility ────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    strat_vol = results['index_level'].pct_change().rolling(30).std() * np.sqrt(252) * 100
    spy_vol   = results['spy_bnh'].pct_change().rolling(30).std()     * np.sqrt(252) * 100
    ax3.plot(results.index, strat_vol, color=color, lw=1.5, label=strategy_name)
    ax3.plot(results.index, spy_vol,   color=BLUE,  lw=1.5, label='SPY', alpha=0.8, ls='--')
    style_ax(ax3, 'Rolling 30d Annualised Volatility (%)')
    legend(ax3)

    # ── Panel 5: Metrics table ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(PANEL); ax5.axis('off')
    m   = compute_metrics(results)
    spy_only = results[['spy_bnh']].rename(columns={'spy_bnh': 'index_level'})
    m_s = compute_metrics(spy_only)

    rows = [
        ('Total Return',    f"{m['Total Return']:.2%}",    f"{m_s['Total Return']:.2%}"),
        ('Ann. Return',     f"{m['Ann. Return']:.2%}",     f"{m_s['Ann. Return']:.2%}"),
        ('Ann. Volatility', f"{m['Ann. Volatility']:.2%}", f"{m_s['Ann. Volatility']:.2%}"),
        ('Sharpe Ratio',    f"{m['Sharpe Ratio']:.3f}",    f"{m_s['Sharpe Ratio']:.3f}"),
        ('Sortino Ratio',   f"{m['Sortino Ratio']:.3f}",   f"{m_s['Sortino Ratio']:.3f}"),
        ('Max Drawdown',    f"{m['Max Drawdown']:.2%}",    f"{m_s['Max Drawdown']:.2%}"),
        ('Calmar Ratio',    f"{m['Calmar Ratio']:.3f}",    f"{m_s['Calmar Ratio']:.3f}"),
        ('VaR 95%',         f"{m['VaR 95%']:.2%}",         f"{m_s['VaR 95%']:.2%}"),
        ('CVaR 95%',        f"{m['CVaR 95%']:.2%}",        f"{m_s['CVaR 95%']:.2%}"),
    ]
    tbl = ax5.table(
        cellText=rows,
        colLabels=['Metric', strategy_name, 'SPY B&H'],
        cellLoc='center', loc='center', bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#1e1e1e' if r == 0 else PANEL)
        cell.set_edgecolor('#333')
        cell.set_text_props(color=color if (c == 1 and r > 0) else
                            BLUE if (c == 2 and r > 0) else WHITE)
    ax5.set_title('Performance Metrics', color=WHITE, fontsize=10, fontweight='bold', pad=8)

    fig.suptitle(f'{strategy_name}  |  SPY 2020–2022  |  Mid Execution',
                 color=WHITE, fontsize=13, fontweight='bold', y=0.998)
    fig.patch.set_facecolor(BG)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"  Chart saved → {save_path}")
    plt.tight_layout()
    plt.show()
    plt.close()


# ── Multi-strategy comparison dashboard ──────────────────────────────

def plot_comparison_dashboard(results_dict: dict, save_path: str = None):
    """
    Full comparison dashboard across all strategies.
    """
    from src.metrics import metrics_table

    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32)

    colors = [STRATEGY_COLORS.get(n, GOLD) for n in results_dict]

    # ── Panel 1: Index levels ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for (name, res), col in zip(results_dict.items(), colors):
        idx_col = 'spy_bnh' if name == 'SPY B&H' else 'index_level'
        ax1.plot(res.index, res[idx_col], color=col, lw=2 if name != 'SPY B&H' else 1.2,
                 label=name, ls='-' if name != 'SPY B&H' else '--',
                 alpha=1.0 if name != 'SPY B&H' else 0.7)
    style_ax(ax1, 'All Strategies vs SPY Buy & Hold  (Base = 100)', fontsize=11)
    legend(ax1, fontsize=9)

    # ── Panel 2: Drawdown ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for (name, res), col in zip(results_dict.items(), colors):
        idx_col = 'spy_bnh' if name == 'SPY B&H' else 'index_level'
        dd = (res[idx_col] / res[idx_col].cummax() - 1) * 100
        ax2.plot(res.index, dd, color=col, lw=1.5, label=name,
                 ls='--' if name == 'SPY B&H' else '-')
    style_ax(ax2, 'Drawdown (%)')
    legend(ax2)

    # ── Panel 3: Rolling vol ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    for (name, res), col in zip(results_dict.items(), colors):
        idx_col = 'spy_bnh' if name == 'SPY B&H' else 'index_level'
        vol = res[idx_col].pct_change().rolling(30).std() * np.sqrt(252) * 100
        ax3.plot(res.index, vol, color=col, lw=1.5, label=name,
                 ls='--' if name == 'SPY B&H' else '-')
    style_ax(ax3, 'Rolling 30d Annualised Volatility (%)')
    legend(ax3)

    # ── Panel 4: Sharpe bar chart ─────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    names, sharpes, bar_cols = [], [], []
    for (name, res), col in zip(results_dict.items(), colors):
        idx_col = 'spy_bnh' if name == 'SPY B&H' else 'index_level'
        r   = res[idx_col].pct_change().dropna()
        ann = (1 + r).prod() ** (252 / len(r)) - 1
        vol = r.std() * np.sqrt(252)
        sh  = ann / vol if vol > 0 else 0
        names.append(name); sharpes.append(sh)
        bar_cols.append(GREEN if sh > 0 else RED)
    bars = ax4.bar(names, sharpes, color=bar_cols, alpha=0.8, edgecolor=GREY)
    for bar, val in zip(bars, sharpes):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01 * np.sign(bar.get_height()),
                 f'{val:.2f}', ha='center', va='bottom', color=WHITE, fontsize=9)
    style_ax(ax4, 'Sharpe Ratio — Comparison')
    ax4.tick_params(axis='x', labelsize=8)

    # ── Panel 5: Metrics table ────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(PANEL); ax5.axis('off')
    mt = metrics_table(results_dict)
    tbl = ax5.table(
        cellText=mt.values,
        rowLabels=mt.index,
        colLabels=mt.columns,
        cellLoc='center', loc='center', bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(7)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#1e1e1e' if r <= 0 else PANEL)
        cell.set_edgecolor('#333')
        cell.set_text_props(color=GOLD if r > 0 else WHITE)
    ax5.set_title('Comparative Performance Metrics', color=WHITE,
                  fontsize=10, fontweight='bold', pad=8)

    fig.suptitle("SPY Options Strategies — Full Comparison  (2020–2022)",
                 color=WHITE, fontsize=13, fontweight='bold', y=0.998)
    fig.patch.set_facecolor(BG)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"  Comparison chart saved → {save_path}")
    plt.tight_layout()
    plt.show()
    plt.close()
