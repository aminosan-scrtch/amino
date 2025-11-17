#!/usr/bin/env python3
"""
プロフェッショナル分析の厳格な検証
Look-ahead Bias、過学習、データリークの徹底チェック

第三者監査の観点で以下を検証:
1. Look-ahead Bias（未来データの取り込み）
2. データリーク（訓練・テストの分離）
3. 過学習の兆候
4. バックテストの現実性
5. 統計的有意性
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 100)
print("🔍 プロフェッショナル分析の厳格な検証（第三者監査）")
print("=" * 100)

# ============================================================================
# 検証1: ラベル生成とバックテストのLook-ahead Bias
# ============================================================================
print("\n[検証1] ラベル生成とバックテストのLook-ahead Bias")
print("-" * 100)

print("\n⚠️ 【重大な問題発見】ラベル生成に未来データの完璧な知識を使用")
print("""
問題のコード:
```python
def calculate_forward_returns(data, index, holding_periods=[4, 12, 24]):
    for period in holding_periods:
        # 将来の最高価格を使用 ← これが問題！
        future_high = data.iloc[index+1:index+1+period]['high'].max()
        gross_return = (future_high - entry_price) / entry_price
```

問題点:
1. 将来の「最高価格」を知っているという前提
2. 実際の取引では不可能（神の視点）
3. これにより勝率が大幅に水増しされる

影響度: ⚡⚡⚡ 致命的
""")

print("\n⚠️ 【重大な問題発見2】バックテストで未来の最高リターンを直接使用")
print("""
問題のコード:
```python
for idx, row in backtest_data.iterrows():
    if row['prediction'] == 1:
        net_return = row['max_net_return']  # ← 将来の最高リターン！
```

問題点:
1. 予測時点で「将来どこまで上がるか」を完璧に知っている前提
2. 実際は利益確定ポイントが不明
3. 勝率84.8%、Sharpe 99.887は過大評価

影響度: ⚡⚡⚡ 致命的
""")

# ============================================================================
# 検証2: データ量と統計的有意性
# ============================================================================
print("\n[検証2] データ量と統計的有意性")
print("-" * 100)

# データ読み込み
df = pd.read_csv('BITSTAMP_BTCUSD, 60.csv')
trades_df = pd.read_csv('backtest_trades.csv')

original_samples = len(df)
sampled_samples = 2494  # スクリプトからの出力
test_trades = len(trades_df)

print(f"元データ: {original_samples:,} 時間足")
print(f"サンプリング後: {sampled_samples:,} サンプル（10時間ごと）")
print(f"削減率: {(1 - sampled_samples/original_samples)*100:.1f}%")
print(f"\n⚠️ 問題: 90%のデータを捨てている")

print(f"\nバックテスト期間のトレード数: {test_trades}")
print(f"⚠️ 問題: サンプル数が少なすぎる（最低100-200トレード推奨）")

# 統計的有意性の計算
from scipy import stats
win_rate = 0.848
n = test_trades
# 二項検定
p_value = stats.binom_test(int(n * win_rate), n, 0.5, alternative='greater')

print(f"\n統計的検定:")
print(f"  帰無仮説: 勝率 = 50%")
print(f"  観測勝率: {win_rate*100:.1f}%")
print(f"  p値: {p_value:.6f}")
print(f"  有意か: {'はい（p<0.05）' if p_value < 0.05 else 'いいえ'}")
print(f"\n⚠️ 注意: サンプル数46は統計的に不十分")

# ============================================================================
# 検証3: 過学習の兆候
# ============================================================================
print("\n[検証3] 過学習（Overfitting）の兆候")
print("-" * 100)

walkforward_df = pd.read_csv('walkforward_results.csv')
montecarlo_df = pd.read_csv('montecarlo_results.csv')

# ウォークフォワード vs バックテスト勝率の比較
wf_lightgbm = walkforward_df[walkforward_df['model'] == 'LightGBM']
wf_avg_win_rate = wf_lightgbm['win_rate'].mean()
backtest_win_rate = 0.848

print(f"ウォークフォワード平均勝率: {wf_avg_win_rate*100:.1f}%")
print(f"最終バックテスト勝率: {backtest_win_rate*100:.1f}%")
print(f"差: {(backtest_win_rate - wf_avg_win_rate)*100:.1f}%ポイント")

if backtest_win_rate > wf_avg_win_rate * 1.2:
    print(f"\n⚡ 警告: バックテストが20%以上良い → 過学習の可能性")
else:
    print(f"\n✓ 許容範囲内")

# モンテカルロとバックテストの比較
mc_median = montecarlo_df['win_rate'].median()
print(f"\nモンテカルロ中央値: {mc_median*100:.1f}%")
print(f"バックテスト勝率: {backtest_win_rate*100:.1f}%")
print(f"差: {(backtest_win_rate - mc_median)*100:.1f}%ポイント")

if backtest_win_rate > mc_median * 1.5:
    print(f"\n⚡⚡ 重大警告: バックテストが50%以上良い → 深刻な問題")

# ============================================================================
# 検証4: バックテストの現実性チェック
# ============================================================================
print("\n[検証4] バックテストの現実性")
print("-" * 100)

trades_df['datetime'] = pd.to_datetime(trades_df['datetime'])
test_period_days = (trades_df['datetime'].max() - trades_df['datetime'].min()).days

print(f"テスト期間: {trades_df['datetime'].min()} ～ {trades_df['datetime'].max()}")
print(f"日数: {test_period_days} 日")
print(f"トレード数: {len(trades_df)}")
print(f"1日あたりトレード: {len(trades_df) / test_period_days:.2f}")

if test_period_days < 90:
    print(f"\n⚠️ 警告: テスト期間が3ヶ月未満 → 不十分")

# 最大ドローダウンの現実性
max_dd = -0.0002  # -0.02%
print(f"\n最大ドローダウン: {max_dd*100:.2f}%")
print(f"⚠️⚠️⚠️ 極めて異常: 0.02%のドローダウンは非現実的")
print(f"理由: 将来の最高価格を知っている前提でトレードしているため")

# Sharpe Ratioの現実性
sharpe = 99.887
print(f"\nSharpe Ratio: {sharpe:.3f}")
print(f"⚠️⚠️⚠️ 極めて異常: 通常のHFTでも2-5程度")
print(f"ヘッジファンドトップでも: 3-4程度")
print(f"この値は理論的にほぼ不可能")

# ============================================================================
# 検証5: 特徴量とラベルの時間的関係
# ============================================================================
print("\n[検証5] 特徴量とラベルの時間的整合性")
print("-" * 100)

print("""
特徴量計算: ✓ Point-in-Timeで正しく実装
  - 過去データのみ使用
  - calculate_features_no_lookahead()で厳格に管理

ラベル生成: ⚡⚡⚡ 重大な問題
  - 将来の最高価格を使用（神の視点）
  - 実取引では不可能な利益確定を前提

バックテスト: ⚡⚡⚡ 重大な問題
  - max_net_returnを直接使用
  - 完璧な利益確定を前提
""")

# ============================================================================
# 検証6: 19連勝の確率分析
# ============================================================================
print("\n[検証6] 19連勝の統計的妥当性")
print("-" * 100)

win_streak = 19
random_prob = (0.5 ** win_streak)
print(f"19連勝が偶然起こる確率（勝率50%と仮定）: {random_prob:.2e}")
print(f"= 約1/{1/random_prob:,.0f}")

# 実際の勝率での確率
actual_win_rate = 0.848
streak_prob_actual = actual_win_rate ** win_streak
print(f"\n勝率84.8%での19連勝確率: {streak_prob_actual:.4f} ({streak_prob_actual*100:.2f}%)")

print(f"\n⚠️ 判定: 勝率84.8%なら19連勝は起こりうる")
print(f"しかし、勝率84.8%自体が過大評価されている可能性が高い")

# ============================================================================
# 検証7: 取引コストの妥当性
# ============================================================================
print("\n[検証7] 取引コストの検証")
print("-" * 100)

TOTAL_COST = 0.004  # 0.4%
avg_return = trades_df['return'].mean()
avg_return_gross = avg_return + TOTAL_COST

print(f"取引コスト（往復）: {TOTAL_COST*100:.2f}%")
print(f"平均リターン（純）: {avg_return*100:.2f}%")
print(f"平均リターン（総）推定: {avg_return_gross*100:.2f}%")

print(f"\n取引コスト設定:")
print(f"  スプレッド: 0.10%")
print(f"  手数料: 0.05%")
print(f"  スリッページ: 0.05%")
print(f"  合計: 0.40% (往復)")
print(f"\n✓ 取引コスト設定は妥当")

# ============================================================================
# 検証8: フォールド間の勝率変動
# ============================================================================
print("\n[検証8] ウォークフォワードフォールド間の安定性")
print("-" * 100)

lightgbm_folds = walkforward_df[walkforward_df['model'] == 'LightGBM'].sort_values('fold')
print("\nLightGBM各フォールドの勝率:")
for _, row in lightgbm_folds.iterrows():
    print(f"  Fold {row['fold']}: {row['win_rate']*100:.1f}%")

win_rates = lightgbm_folds['win_rate'].values
std_dev = np.std(win_rates)
coef_var = std_dev / np.mean(win_rates)

print(f"\n標準偏差: {std_dev:.4f}")
print(f"変動係数: {coef_var:.4f}")

if coef_var > 0.1:
    print(f"⚠️ 警告: 変動が大きい（不安定）")
else:
    print(f"✓ 安定している")

# ============================================================================
# 総合判定
# ============================================================================
print("\n" + "=" * 100)
print("📋 総合判定レポート")
print("=" * 100)

issues = []

# 致命的問題
critical_issues = [
    "⚡⚡⚡ CRITICAL: ラベル生成で将来の最高価格を使用（Look-ahead Bias）",
    "⚡⚡⚡ CRITICAL: バックテストでmax_net_returnを直接使用（非現実的）",
    "⚡⚡ HIGH: バックテストサンプル数が46と少なすぎる",
    "⚡⚡ HIGH: Sharpe Ratio 99.887は非現実的（通常2-5程度）",
    "⚡⚡ HIGH: 最大ドローダウン0.02%は異常に小さい",
]

# 警告レベルの問題
warning_issues = [
    "⚠️ WARNING: 10時間ごとのサンプリングで90%のデータを削減",
    "⚠️ WARNING: テスト期間が3.5ヶ月と短い",
    "⚠️ WARNING: ウォークフォワード勝率60.5% vs バックテスト84.8%の乖離",
]

print("\n【致命的問題】")
for issue in critical_issues:
    print(f"  {issue}")

print("\n【警告レベルの問題】")
for issue in warning_issues:
    print(f"  {issue}")

print("\n" + "=" * 100)
print("🔴 最終結論")
print("=" * 100)

print("""
この分析結果は以下の理由により【信頼できません】:

1. 【根本的な問題】ラベル生成の欠陥
   - 将来の「最高価格」を知っている前提でラベル付け
   - これは実取引では不可能な「神の視点」
   - 勝率84.8%は大幅に過大評価

2. 【バックテストの問題】
   - max_net_return（将来の最高リターン）を直接使用
   - 完璧な利益確定を前提
   - Sharpe 99.887、DD 0.02%は非現実的

3. 【統計的問題】
   - テストサンプル数46は不十分
   - 期間3.5ヶ月は短すぎ

4. 【過学習の兆候】
   - ウォークフォワード60.5% → バックテスト84.8%
   - 大幅な乖離は過学習を示唆

【正しい結果の推定】
- 実際の勝率: 50-55%程度（モンテカルロ結果に近い）
- Sharpe Ratio: 1.5-3.0程度
- 最大ドローダウン: 5-15%程度

【修正が必要な箇所】
1. ラベル生成: 固定期間（例：24時間）後の価格を使用
2. バックテスト: 実際の利益確定ルール（固定期間または損切り）
3. サンプル数: 全データを使用（サンプリングなし）
4. テスト期間: 最低6-12ヶ月
""")

print("\n" + "=" * 100)
print("次のステップ: 修正版スクリプトの作成")
print("=" * 100)
