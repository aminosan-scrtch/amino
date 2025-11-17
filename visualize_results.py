#!/usr/bin/env python3
"""
プロフェッショナル分析結果の可視化
図表を生成してMarkdownレポートを作成
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUIなし環境用
import seaborn as sns
from datetime import datetime

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# スタイル設定
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 100

print("=" * 100)
print("プロフェッショナル分析結果の可視化")
print("=" * 100)

# ============================================================================
# データ読み込み
# ============================================================================
print("\n[1/7] データ読み込み中...")

walkforward_df = pd.read_csv('walkforward_results.csv')
montecarlo_df = pd.read_csv('montecarlo_results.csv')
trades_df = pd.read_csv('backtest_trades.csv')
shap_df = pd.read_csv('shap_feature_importance.csv')

print(f"  ウォークフォワード結果: {len(walkforward_df)} 行")
print(f"  モンテカルロ結果: {len(montecarlo_df)} 行")
print(f"  トレード履歴: {len(trades_df)} 行")
print(f"  SHAP特徴量: {len(shap_df)} 行")

# ============================================================================
# 図1: ウォークフォワード分析 - モデル別勝率比較
# ============================================================================
print("\n[2/7] 図1: モデル別勝率比較グラフ生成中...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# モデル別の平均勝率
model_performance = walkforward_df.groupby('model')['win_rate'].agg(['mean', 'std']).sort_values('mean', ascending=False)

ax1.barh(model_performance.index, model_performance['mean'], xerr=model_performance['std'],
         color='skyblue', edgecolor='navy', alpha=0.7, capsize=5)
ax1.set_xlabel('Average Win Rate', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Comparison (Walk-Forward Analysis)', fontsize=14, fontweight='bold')
ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Baseline (50%)')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(model_performance.iterrows()):
    ax1.text(row['mean'] + 0.01, i, f"{row['mean']:.1%}", va='center', fontsize=10)

# フォールド別勝率推移
for model in walkforward_df['model'].unique():
    model_data = walkforward_df[walkforward_df['model'] == model]
    ax2.plot(model_data['fold'], model_data['win_rate'], marker='o', label=model, linewidth=2)

ax2.set_xlabel('Walk-Forward Fold', fontsize=12, fontweight='bold')
ax2.set_ylabel('Win Rate', fontsize=12, fontweight='bold')
ax2.set_title('Win Rate Evolution Across Folds', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(alpha=0.3)
ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('fig1_model_comparison.png', dpi=150, bbox_inches='tight')
print("  ✓ 保存: fig1_model_comparison.png")
plt.close()

# ============================================================================
# 図2: モンテカルロシミュレーション結果の分布
# ============================================================================
print("\n[3/7] 図2: モンテカルロシミュレーション結果生成中...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ヒストグラム
ax1.hist(montecarlo_df['win_rate'], bins=30, color='lightcoral', edgecolor='darkred', alpha=0.7)
ax1.axvline(montecarlo_df['win_rate'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: {montecarlo_df["win_rate"].median():.1%}')
ax1.axvline(montecarlo_df['win_rate'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {montecarlo_df["win_rate"].mean():.1%}')
ax1.set_xlabel('Win Rate', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Monte Carlo Simulation: Win Rate Distribution (100 Trials)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# ボックスプロット
box_data = montecarlo_df['win_rate']
bp = ax2.boxplot([box_data], vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightgreen')
bp['boxes'][0].set_edgecolor('darkgreen')
bp['medians'][0].set_color('red')
bp['medians'][0].set_linewidth(2)

# 分位点の注釈
percentiles = [5, 25, 50, 75, 95]
percentile_values = [np.percentile(box_data, p) for p in percentiles]
for p, val in zip(percentiles, percentile_values):
    ax2.text(1.2, val, f'{p}%: {val:.1%}', va='center', fontsize=10, fontweight='bold')

ax2.set_ylabel('Win Rate', fontsize=12, fontweight='bold')
ax2.set_title('Monte Carlo Results: Robustness Analysis', fontsize=14, fontweight='bold')
ax2.set_xticklabels(['Parameter Noise ±20%'])
ax2.grid(axis='y', alpha=0.3)

# 統計情報
stats_text = f'Std Dev: {box_data.std():.3f}\nRange: {box_data.max()-box_data.min():.3f}'
ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('fig2_montecarlo_distribution.png', dpi=150, bbox_inches='tight')
print("  ✓ 保存: fig2_montecarlo_distribution.png")
plt.close()

# ============================================================================
# 図3: エクイティカーブ（資産推移）とドローダウン
# ============================================================================
print("\n[4/7] 図3: エクイティカーブとドローダウン生成中...")

trades_df['datetime'] = pd.to_datetime(trades_df['datetime'])
trades_df = trades_df.sort_values('datetime')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# エクイティカーブ
ax1.plot(trades_df['datetime'], trades_df['equity'], linewidth=2, color='darkgreen', label='Equity')
ax1.fill_between(trades_df['datetime'], 1.0, trades_df['equity'], alpha=0.3, color='lightgreen')
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
ax1.set_ylabel('Equity (Normalized)', fontsize=12, fontweight='bold')
ax1.set_title('Equity Curve: Capital Growth Over Time', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# 統計情報
total_return = (trades_df['equity'].iloc[-1] - 1.0) * 100
ax1.text(0.02, 0.98, f'Total Return: +{total_return:.2f}%\nTrades: {len(trades_df)}',
         transform=ax1.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# ドローダウン
equity_series = trades_df['equity'].values
running_max = np.maximum.accumulate(equity_series)
drawdowns = (equity_series - running_max) / running_max * 100

ax2.fill_between(trades_df['datetime'], 0, drawdowns, color='red', alpha=0.3, label='Drawdown')
ax2.plot(trades_df['datetime'], drawdowns, linewidth=2, color='darkred')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
ax2.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

max_dd = drawdowns.min()
ax2.text(0.02, 0.02, f'Max Drawdown: {max_dd:.2f}%',
         transform=ax2.transAxes, fontsize=11,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('fig3_equity_drawdown.png', dpi=150, bbox_inches='tight')
print("  ✓ 保存: fig3_equity_drawdown.png")
plt.close()

# ============================================================================
# 図4: トレード分析（勝ち負けの分布）
# ============================================================================
print("\n[5/7] 図4: トレード分析生成中...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# リターン分布
winning_trades = trades_df[trades_df['return'] > 0]['return'] * 100
losing_trades = trades_df[trades_df['return'] <= 0]['return'] * 100

ax1.hist([winning_trades, losing_trades], bins=20, label=['Winning Trades', 'Losing Trades'],
         color=['green', 'red'], alpha=0.7, edgecolor='black')
ax1.set_xlabel('Return (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Trade Return Distribution', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax1.grid(alpha=0.3)

# 統計情報
win_rate = len(winning_trades) / len(trades_df) * 100
avg_win = winning_trades.mean()
avg_loss = losing_trades.mean()
ax1.text(0.98, 0.98, f'Win Rate: {win_rate:.1f}%\nAvg Win: +{avg_win:.2f}%\nAvg Loss: {avg_loss:.2f}%',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# P&L推移
trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum() * 100
ax2.plot(trades_df['datetime'], trades_df['cumulative_pnl'], linewidth=2, color='purple', marker='o', markersize=3)
ax2.fill_between(trades_df['datetime'], 0, trades_df['cumulative_pnl'], alpha=0.3, color='purple')
ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cumulative P&L (%)', fontsize=12, fontweight='bold')
ax2.set_title('Cumulative Profit & Loss', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# トレードごとのP&L
colors = ['green' if p > 0 else 'red' for p in trades_df['pnl']]
ax3.bar(range(len(trades_df)), trades_df['pnl'] * 100, color=colors, alpha=0.7, edgecolor='black')
ax3.set_xlabel('Trade Number', fontsize=12, fontweight='bold')
ax3.set_ylabel('P&L (%)', fontsize=12, fontweight='bold')
ax3.set_title('Individual Trade Performance', fontsize=14, fontweight='bold')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax3.grid(axis='y', alpha=0.3)

# 連勝・連敗分析
trades_df['win'] = (trades_df['return'] > 0).astype(int)
streak_changes = trades_df['win'].diff().fillna(0) != 0
streak_groups = streak_changes.cumsum()
streaks = trades_df.groupby(streak_groups).agg({
    'win': ['first', 'count']
})
streaks.columns = ['is_win', 'length']

win_streaks = streaks[streaks['is_win'] == 1]['length']
loss_streaks = streaks[streaks['is_win'] == 0]['length']

ax4.hist([win_streaks, loss_streaks], bins=range(1, max(win_streaks.max(), loss_streaks.max()) + 2),
         label=['Win Streaks', 'Loss Streaks'], color=['green', 'red'], alpha=0.7, edgecolor='black', align='left')
ax4.set_xlabel('Streak Length', fontsize=12, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax4.set_title('Consecutive Win/Loss Analysis', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(alpha=0.3)

# 統計
max_win_streak = win_streaks.max() if len(win_streaks) > 0 else 0
max_loss_streak = loss_streaks.max() if len(loss_streaks) > 0 else 0
ax4.text(0.98, 0.98, f'Max Win Streak: {max_win_streak}\nMax Loss Streak: {max_loss_streak}',
         transform=ax4.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.savefig('fig4_trade_analysis.png', dpi=150, bbox_inches='tight')
print("  ✓ 保存: fig4_trade_analysis.png")
plt.close()

# ============================================================================
# 図5: SHAP値による特徴量重要度
# ============================================================================
print("\n[6/7] 図5: 特徴量重要度グラフ生成中...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# TOP 15特徴量
top_features = shap_df.head(15).sort_values('importance')

ax1.barh(top_features['feature'], top_features['importance'], color='steelblue', edgecolor='navy', alpha=0.8)
ax1.set_xlabel('SHAP Importance Value', fontsize=12, fontweight='bold')
ax1.set_title('Top 15 Feature Importance (SHAP Values)', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(top_features.iterrows()):
    ax1.text(row['importance'] + 0.002, i, f"{row['importance']:.4f}", va='center', fontsize=9)

# 特徴量カテゴリ別集計
def categorize_feature(name):
    if 'volatility' in name or 'vol_ratio' in name:
        return 'Volatility'
    elif 'MA_' in name or 'trend' in name:
        return 'Trend/MA'
    elif 'RSI' in name or 'MACD' in name or 'BB_' in name:
        return 'Technical'
    elif 'momentum' in name:
        return 'Momentum'
    elif 'day_' in name or 'hour_' in name:
        return 'Temporal'
    elif 'range' in name or 'position' in name:
        return 'Price Structure'
    else:
        return 'Other'

shap_df['category'] = shap_df['feature'].apply(categorize_feature)
category_importance = shap_df.groupby('category')['importance'].sum().sort_values(ascending=False)

colors_cat = plt.cm.Set3(range(len(category_importance)))
wedges, texts, autotexts = ax2.pie(category_importance, labels=category_importance.index, autopct='%1.1f%%',
                                     colors=colors_cat, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})

ax2.set_title('Feature Importance by Category', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('fig5_feature_importance.png', dpi=150, bbox_inches='tight')
print("  ✓ 保存: fig5_feature_importance.png")
plt.close()

# ============================================================================
# 表の生成
# ============================================================================
print("\n[7/7] 表の生成中...")

# 表1: モデル性能サマリー
model_summary = walkforward_df.groupby('model').agg({
    'win_rate': ['mean', 'std', 'min', 'max'],
    'accuracy': 'mean',
    'n_signals': 'mean'
}).round(4)
model_summary.to_csv('table1_model_summary.csv')
print("  ✓ 保存: table1_model_summary.csv")

# 表2: トレード統計
trade_stats = {
    'Metric': [
        'Total Trades',
        'Winning Trades',
        'Losing Trades',
        'Win Rate (%)',
        'Average Win (%)',
        'Average Loss (%)',
        'Profit Factor',
        'Total Return (%)',
        'Max Drawdown (%)',
        'Largest Win (%)',
        'Largest Loss (%)'
    ],
    'Value': [
        len(trades_df),
        len(winning_trades),
        len(losing_trades),
        f"{win_rate:.2f}",
        f"{avg_win:.2f}",
        f"{avg_loss:.2f}",
        f"{abs(avg_win / avg_loss):.2f}",
        f"{total_return:.2f}",
        f"{max_dd:.2f}",
        f"{winning_trades.max():.2f}",
        f"{losing_trades.min():.2f}"
    ]
}
pd.DataFrame(trade_stats).to_csv('table2_trade_statistics.csv', index=False)
print("  ✓ 保存: table2_trade_statistics.csv")

# 表3: リスク指標
returns_series = trades_df['return'].values
sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(365 * 24) if returns_series.std() > 0 else 0
calmar = (total_return / 100) / abs(max_dd / 100) if max_dd != 0 else 0

risk_metrics = {
    'Metric': [
        'Sharpe Ratio',
        'Calmar Ratio',
        'Sortino Ratio',
        'Max Consecutive Wins',
        'Max Consecutive Losses',
        'Average Trade Duration (hours)',
        'Best Month Return (%)',
        'Worst Month Return (%)'
    ],
    'Value': [
        f"{sharpe:.3f}",
        f"{calmar:.3f}",
        'N/A',
        f"{max_win_streak}",
        f"{max_loss_streak}",
        'N/A',
        'N/A',
        'N/A'
    ]
}
pd.DataFrame(risk_metrics).to_csv('table3_risk_metrics.csv', index=False)
print("  ✓ 保存: table3_risk_metrics.csv")

print("\n" + "=" * 100)
print("可視化完了！")
print("=" * 100)
print("\n生成されたファイル:")
print("  図:")
print("    - fig1_model_comparison.png")
print("    - fig2_montecarlo_distribution.png")
print("    - fig3_equity_drawdown.png")
print("    - fig4_trade_analysis.png")
print("    - fig5_feature_importance.png")
print("  表:")
print("    - table1_model_summary.csv")
print("    - table2_trade_statistics.csv")
print("    - table3_risk_metrics.csv")
