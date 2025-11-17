#!/usr/bin/env python3
"""
修正版: 現実的なBTC/USD機械学習投資分析
致命的欠陥を修正し、実取引可能な戦略を構築

修正点:
1. ラベル生成: 固定24時間保有 + 利確/損切りルール
2. バックテスト: 現実的な利確ロジック
3. サンプリング: 全データ使用（または4時間ごと）
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import lightgbm as lgb

print("=" * 100)
print("修正版 BTC/USD 機械学習投資分析（現実的バージョン）")
print("=" * 100)

# ============================================================================
# 1. データ読み込み
# ============================================================================
print("\n[1/8] データ読み込み")
print("-" * 100)

df = pd.read_csv('BITSTAMP_BTCUSD, 60.csv')
df['datetime'] = pd.to_datetime(df['time'], unit='s')
df = df.sort_values('datetime').reset_index(drop=True)

print(f"総データ数: {len(df):,} 時間足")
print(f"期間: {df['datetime'].min()} ～ {df['datetime'].max()}")

# ============================================================================
# 2. 取引コスト設定
# ============================================================================
TRADING_COSTS = {
    'spread_pct': 0.001,
    'commission_pct': 0.0005,
    'slippage_pct': 0.0005
}
TOTAL_COST_PCT = sum(TRADING_COSTS.values()) * 2  # 往復

# ============================================================================
# 3. 修正版ラベル生成（現実的）
# ============================================================================
print("\n[2/8] 修正版ラベル生成（固定保有期間 + 利確/損切り）")
print("-" * 100)

def calculate_realistic_return(data, index,
                               holding_period=24,
                               take_profit=0.02,  # +2%で利確
                               stop_loss=-0.01):  # -1%で損切り
    """
    現実的なリターン計算
    - 固定保有期間内で利確/損切りルールを適用
    - 将来の最高価格は使用しない
    """
    if index + holding_period >= len(data):
        return None

    entry_price = data.iloc[index]['close']

    # 保有期間中の価格推移をチェック
    for h in range(1, holding_period + 1):
        if index + h >= len(data):
            break

        current_price = data.iloc[index + h]['close']
        return_rate = (current_price - entry_price) / entry_price

        # 利確条件
        if return_rate >= take_profit:
            return return_rate - TOTAL_COST_PCT

        # 損切り条件
        if return_rate <= stop_loss:
            return return_rate - TOTAL_COST_PCT

    # 保有期間終了時の価格
    exit_price = data.iloc[index + holding_period]['close']
    return_rate = (exit_price - entry_price) / entry_price
    return return_rate - TOTAL_COST_PCT

print("ルール:")
print(f"  保有期間: 24時間")
print(f"  利確: +2%到達で即座に決済")
print(f"  損切り: -1%到達で即座に決済")
print(f"  取引コスト: {TOTAL_COST_PCT*100:.2f}%（往復）")

# ============================================================================
# 4. Point-in-Time特徴量計算
# ============================================================================
print("\n[3/8] Point-in-Time特徴量計算")
print("-" * 100)

def calculate_features_no_lookahead(data, index):
    """過去データのみを使用した特徴量計算"""
    hist_data = data.iloc[:index+1].copy()

    if len(hist_data) < 100:
        return None

    features = {}
    current = hist_data.iloc[-1]
    close_series = hist_data['close']

    # 移動平均
    features['MA_5'] = close_series.iloc[-5:].mean()
    features['MA_20'] = close_series.iloc[-20:].mean()
    features['MA_50'] = close_series.iloc[-50:].mean()
    if len(hist_data) >= 100:
        features['MA_100'] = close_series.iloc[-100:].mean()
    else:
        features['MA_100'] = close_series.mean()

    # ボラティリティ
    returns = close_series.pct_change().dropna()
    features['volatility_20'] = returns.iloc[-20:].std() * np.sqrt(24 * 365)
    features['volatility_50'] = returns.iloc[-50:].std() * np.sqrt(24 * 365)

    # RSI
    def calc_rsi(prices, period=14):
        deltas = prices.diff()
        gain = deltas.where(deltas > 0, 0).iloc[-period:].mean()
        loss = -deltas.where(deltas < 0, 0).iloc[-period:].mean()
        if loss == 0:
            return 100
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    features['RSI_14'] = calc_rsi(close_series, 14)

    # MACD
    ema_12 = close_series.ewm(span=12).mean().iloc[-1]
    ema_26 = close_series.ewm(span=26).mean().iloc[-1]
    macd = ema_12 - ema_26
    macd_series = close_series.ewm(span=12).mean() - close_series.ewm(span=26).mean()
    signal = macd_series.ewm(span=9).mean().iloc[-1]
    features['MACD_diff'] = macd - signal

    # ボリンジャーバンド
    bb_middle = close_series.iloc[-20:].mean()
    bb_std = close_series.iloc[-20:].std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    features['BB_position'] = (current['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # モメンタム
    if len(hist_data) >= 25:
        features['momentum_24h'] = (current['close'] / hist_data.iloc[-25]['close'] - 1)
    else:
        features['momentum_24h'] = 0

    # トレンド
    features['trend_20'] = (current['close'] / features['MA_20'] - 1)
    features['trend_50'] = (current['close'] / features['MA_50'] - 1)

    # 時刻特徴
    hour = current['datetime'].hour if 'datetime' in current else 0
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    dayofweek = current['datetime'].dayofweek if 'datetime' in current else 0
    features['day_sin'] = np.sin(2 * np.pi * dayofweek / 7)
    features['day_cos'] = np.cos(2 * np.pi * dayofweek / 7)

    return features

# ============================================================================
# 5. データセット構築（サンプリング削減）
# ============================================================================
print("\n[4/8] データセット構築")
print("-" * 100)

dataset = []
feature_names = None

max_index = len(df) - 24
sampling_interval = 4  # 4時間ごと（計算量とデータ量のバランス）

print(f"サンプリング間隔: {sampling_interval}時間ごと")
print("処理中...")

for i in range(100, max_index, sampling_interval):
    if i % 2000 == 0:
        print(f"  進捗: {i}/{max_index} ({i/max_index*100:.1f}%)")

    features = calculate_features_no_lookahead(df, i)
    if features is None:
        continue

    # 修正版リターン計算
    net_return = calculate_realistic_return(df, i)
    if net_return is None:
        continue

    datapoint = {**features}
    datapoint['net_return'] = net_return
    datapoint['index'] = i
    datapoint['datetime'] = df.iloc[i]['datetime']
    datapoint['close'] = df.iloc[i]['close']

    dataset.append(datapoint)

    if feature_names is None:
        feature_names = [k for k in features.keys()]

dataset_df = pd.DataFrame(dataset)
print(f"\nデータセット: {len(dataset_df):,} サンプル")
print(f"特徴量数: {len(feature_names)}")

# ラベル: 1%以上の利益
PROFIT_THRESHOLD = 0.01
dataset_df['target'] = (dataset_df['net_return'] > PROFIT_THRESHOLD).astype(int)

print(f"\n買いシグナル: {dataset_df['target'].sum():,} ({dataset_df['target'].mean()*100:.1f}%)")
print(f"見送り: {(1-dataset_df['target']).sum():,}")

# ============================================================================
# 6. ウォークフォワード分析
# ============================================================================
print("\n[5/8] ウォークフォワード分析")
print("-" * 100)

# 時系列分割（80%訓練、20%テスト）
split_idx = int(len(dataset_df) * 0.8)
train_data = dataset_df.iloc[:split_idx]
test_data = dataset_df.iloc[split_idx:]

X_train = train_data[feature_names]
y_train = train_data['target']
X_test = test_data[feature_names]
y_test = test_data['target']

print(f"訓練: {len(X_train):,} サンプル")
print(f"テスト: {len(X_test):,} サンプル")
print(f"訓練期間: {train_data['datetime'].min()} ～ {train_data['datetime'].max()}")
print(f"テスト期間: {test_data['datetime'].min()} ～ {test_data['datetime'].max()}")

# ============================================================================
# 7. モデル訓練
# ============================================================================
print("\n[6/8] LightGBMモデル訓練")
print("-" * 100)

model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

# 勝率計算
train_buy = y_pred_train == 1
test_buy = y_pred_test == 1

train_win_rate = y_train.values[train_buy].mean() if train_buy.sum() > 0 else 0
test_win_rate = y_test.values[test_buy].mean() if test_buy.sum() > 0 else 0

print(f"訓練精度: {train_acc:.3f}")
print(f"テスト精度: {test_acc:.3f}")
print(f"訓練買いシグナル勝率: {train_win_rate*100:.1f}%")
print(f"テスト買いシグナル勝率: {test_win_rate*100:.1f}%")
print(f"テスト買いシグナル数: {test_buy.sum()}")

# ============================================================================
# 8. 現実的バックテスト
# ============================================================================
print("\n[7/8] 現実的バックテスト（利確/損切りルール適用）")
print("-" * 100)

backtest_data = test_data.copy()
backtest_data['prediction'] = y_pred_test

trades = []
current_equity = 1.0

for idx, row in backtest_data.iterrows():
    if row['prediction'] == 1:
        # エントリー
        entry_idx = row['index']
        entry_price = row['close']

        # 利確/損切りルールに従ってリターン計算
        actual_return = calculate_realistic_return(df, entry_idx)

        if actual_return is not None:
            position_size = 0.02 * current_equity
            pnl = position_size * actual_return
            current_equity += pnl

            trades.append({
                'datetime': row['datetime'],
                'entry_price': entry_price,
                'return': actual_return,
                'pnl': pnl,
                'equity': current_equity
            })

if trades:
    trades_df = pd.DataFrame(trades)

    total_return = (current_equity - 1.0)
    n_trades = len(trades)
    winning_trades = trades_df[trades_df['return'] > 0]
    losing_trades = trades_df[trades_df['return'] <= 0]

    win_rate = len(winning_trades) / n_trades
    avg_win = winning_trades['return'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['return'].mean() if len(losing_trades) > 0 else 0

    # Sharpe Ratio
    returns_series = trades_df['return'].values
    mean_return = np.mean(returns_series)
    std_return = np.std(returns_series)
    sharpe_ratio = (mean_return / std_return) * np.sqrt(365 * 24) if std_return > 0 else 0

    # 最大ドローダウン
    equity_series = trades_df['equity'].values
    running_max = np.maximum.accumulate(equity_series)
    drawdowns = (equity_series - running_max) / running_max
    max_drawdown = np.min(drawdowns)

    print("\n" + "=" * 100)
    print("バックテスト結果（修正版・現実的）")
    print("=" * 100)
    print(f"総トレード数: {n_trades}")
    print(f"勝率: {win_rate*100:.1f}%")
    print(f"平均勝ち: +{avg_win*100:.2f}%")
    print(f"平均負け: {avg_loss*100:.2f}%")
    print(f"Profit Factor: {abs(avg_win / avg_loss) if avg_loss != 0 else 0:.2f}")
    print(f"総リターン: {total_return*100:.2f}%")
    print(f"最大ドローダウン: {max_drawdown*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.3f}")

    # 保存
    trades_df.to_csv('corrected_backtest_trades.csv', index=False)
    print("\n✓ 結果を 'corrected_backtest_trades.csv' に保存")

else:
    print("トレードが発生しませんでした")

# ============================================================================
# 8. 改善の余地分析
# ============================================================================
print("\n[8/8] 改善の余地分析")
print("-" * 100)

# 特徴量重要度
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n特徴量重要度 TOP 10:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:20s}: {row['importance']:.4f}")

feature_importance.to_csv('corrected_feature_importance.csv', index=False)

print("\n" + "=" * 100)
print("修正版分析完了")
print("=" * 100)
print("\n次のステップ: 改善策の検討")
print("  1. 特徴量追加（オンチェーンデータ、センチメント等）")
print("  2. ハイパーパラメータ最適化")
print("  3. アンサンブル手法")
print("  4. より長いヒストリカルデータ")
