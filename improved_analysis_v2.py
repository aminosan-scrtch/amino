#!/usr/bin/env python3
"""
改善版V2: 高度な特徴量を追加
- 高度なテクニカル指標（ATR, ADX, Stochastic等）
- マルチタイムフレーム
- より洗練された特徴量

期待: 勝率46% → 52-56%
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

print("=" * 100)
print("改善版V2: 高度な特徴量追加")
print("=" * 100)

# ============================================================================
# データ読み込み
# ============================================================================
df = pd.read_csv('BITSTAMP_BTCUSD, 60.csv')
df['datetime'] = pd.to_datetime(df['time'], unit='s')
df = df.sort_values('datetime').reset_index(drop=True)

TRADING_COSTS = {'spread_pct': 0.001, 'commission_pct': 0.0005, 'slippage_pct': 0.0005}
TOTAL_COST_PCT = sum(TRADING_COSTS.values()) * 2

# ============================================================================
# ラベル生成
# ============================================================================
def calculate_realistic_return(data, index, holding_period=24, take_profit=0.02, stop_loss=-0.01):
    if index + holding_period >= len(data):
        return None
    entry_price = data.iloc[index]['close']
    for h in range(1, holding_period + 1):
        if index + h >= len(data):
            break
        current_price = data.iloc[index + h]['close']
        return_rate = (current_price - entry_price) / entry_price
        if return_rate >= take_profit:
            return return_rate - TOTAL_COST_PCT
        if return_rate <= stop_loss:
            return return_rate - TOTAL_COST_PCT
    exit_price = data.iloc[index + holding_period]['close']
    return_rate = (exit_price - entry_price) / entry_price
    return return_rate - TOTAL_COST_PCT

# ============================================================================
# 🆕 改善版特徴量計算（高度なテクニカル指標を大量追加）
# ============================================================================
print("\n[1/6] 改善版特徴量計算（50+ features）")
print("-" * 100)

def calculate_advanced_features(data, index):
    """高度な特徴量を多数追加"""
    hist_data = data.iloc[:index+1].copy()
    if len(hist_data) < 100:
        return None

    features = {}
    current = hist_data.iloc[-1]
    close_series = hist_data['close']
    high_series = hist_data['high']
    low_series = hist_data['low']

    # === 基本的な移動平均 ===
    for period in [5, 10, 20, 50, 100]:
        if len(close_series) >= period:
            features[f'MA_{period}'] = close_series.iloc[-period:].mean()
        else:
            features[f'MA_{period}'] = close_series.mean()

    # === EMA（指数移動平均） ===
    for period in [12, 26, 50]:
        features[f'EMA_{period}'] = close_series.ewm(span=period).mean().iloc[-1]

    # === ボラティリティ ===
    returns = close_series.pct_change().dropna()
    for period in [10, 20, 50]:
        if len(returns) >= period:
            features[f'volatility_{period}'] = returns.iloc[-period:].std() * np.sqrt(24 * 365)

    # === 🆕 ATR (Average True Range) ===
    high_low = high_series - low_series
    high_close = np.abs(high_series - close_series.shift())
    low_close = np.abs(low_series - close_series.shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    for period in [14, 28]:
        if len(true_range) >= period:
            features[f'ATR_{period}'] = true_range.iloc[-period:].mean()

    # === 🆕 ADX (Average Directional Index) - トレンド強度 ===
    def calc_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return 0
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = true_range.iloc[-period:]
        plus_di = 100 * (plus_dm.iloc[-period:].ewm(span=period).mean() / tr.ewm(span=period).mean())
        minus_di = 100 * (minus_dm.iloc[-period:].ewm(span=period).mean() / tr.ewm(span=period).mean())
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        return dx.ewm(span=period).mean().iloc[-1]

    features['ADX_14'] = calc_adx(high_series, low_series, close_series, 14)

    # === RSI（複数期間） ===
    def calc_rsi(prices, period=14):
        deltas = prices.diff()
        gain = deltas.where(deltas > 0, 0).iloc[-period:].mean()
        loss = -deltas.where(deltas < 0, 0).iloc[-period:].mean()
        if loss == 0:
            return 100
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    for period in [7, 14, 28]:
        features[f'RSI_{period}'] = calc_rsi(close_series, period)

    # === 🆕 Stochastic Oscillator ===
    def calc_stochastic(high, low, close, k_period=14, d_period=3):
        if len(close) < k_period:
            return 50, 50
        lowest_low = low.iloc[-k_period:].min()
        highest_high = high.iloc[-k_period:].max()
        k = 100 * (close.iloc[-1] - lowest_low) / (highest_high - lowest_low + 1e-10)
        # %Dは%Kの移動平均
        k_values = []
        for i in range(max(0, len(close) - k_period - d_period), len(close)):
            if i + k_period > len(close):
                break
            ll = low.iloc[i:i+k_period].min()
            hh = high.iloc[i:i+k_period].max()
            k_val = 100 * (close.iloc[i+k_period-1] - ll) / (hh - ll + 1e-10)
            k_values.append(k_val)
        d = np.mean(k_values[-d_period:]) if len(k_values) >= d_period else k
        return k, d

    stoch_k, stoch_d = calc_stochastic(high_series, low_series, close_series)
    features['Stochastic_K'] = stoch_k
    features['Stochastic_D'] = stoch_d

    # === 🆕 CCI (Commodity Channel Index) ===
    def calc_cci(high, low, close, period=20):
        if len(close) < period:
            return 0
        tp = (high + low + close) / 3
        sma_tp = tp.iloc[-period:].mean()
        mad = np.abs(tp.iloc[-period:] - sma_tp).mean()
        return (tp.iloc[-1] - sma_tp) / (0.015 * mad + 1e-10)

    features['CCI_20'] = calc_cci(high_series, low_series, close_series, 20)

    # === 🆕 Williams %R ===
    def calc_williams_r(high, low, close, period=14):
        if len(close) < period:
            return -50
        highest_high = high.iloc[-period:].max()
        lowest_low = low.iloc[-period:].min()
        return -100 * (highest_high - close.iloc[-1]) / (highest_high - lowest_low + 1e-10)

    features['Williams_R'] = calc_williams_r(high_series, low_series, close_series, 14)

    # === MACD（複数設定） ===
    ema_12 = close_series.ewm(span=12).mean().iloc[-1]
    ema_26 = close_series.ewm(span=26).mean().iloc[-1]
    macd = ema_12 - ema_26
    macd_series = close_series.ewm(span=12).mean() - close_series.ewm(span=26).mean()
    signal = macd_series.ewm(span=9).mean().iloc[-1]
    features['MACD'] = macd
    features['MACD_signal'] = signal
    features['MACD_diff'] = macd - signal

    # === ボリンジャーバンド ===
    for period in [20, 50]:
        bb_middle = close_series.iloc[-period:].mean()
        bb_std = close_series.iloc[-period:].std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        features[f'BB_position_{period}'] = (current['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        features[f'BB_width_{period}'] = (bb_upper - bb_lower) / bb_middle

    # === モメンタム（複数期間） ===
    for hours in [4, 12, 24, 48, 72]:
        if len(hist_data) >= hours + 1:
            features[f'momentum_{hours}h'] = (current['close'] / hist_data.iloc[-(hours+1)]['close'] - 1)

    # === トレンド指標 ===
    for period in [5, 20, 50, 100]:
        ma = features.get(f'MA_{period}', current['close'])
        features[f'trend_{period}'] = (current['close'] / ma - 1)

    # === 🆕 マルチタイムフレーム（4時間足の情報） ===
    # 4時間足相当の移動平均
    if len(close_series) >= 80:  # 4時間 * 20 = 80時間
        features['MA_20_4h'] = close_series.iloc[-80::4].mean()  # 4時間ごとにサンプリング

    # === 🆕 Price Action指標 ===
    # 高値・安値レンジ
    features['hl_range'] = current['high'] - current['low']
    features['hl_range_pct'] = features['hl_range'] / current['close']
    features['close_position'] = (current['close'] - current['low']) / (features['hl_range'] + 1e-10)

    # ローソク足の形状
    body = current['close'] - current['open']
    features['candle_body'] = body / current['close']
    upper_shadow = current['high'] - max(current['open'], current['close'])
    lower_shadow = min(current['open'], current['close']) - current['low']
    features['upper_shadow'] = upper_shadow / current['close']
    features['lower_shadow'] = lower_shadow / current['close']

    # === 時刻特徴 ===
    hour = current['datetime'].hour if 'datetime' in current else 0
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    dayofweek = current['datetime'].dayofweek if 'datetime' in current else 0
    features['day_sin'] = np.sin(2 * np.pi * dayofweek / 7)
    features['day_cos'] = np.cos(2 * np.pi * dayofweek / 7)

    return features

# ============================================================================
# データセット構築
# ============================================================================
print("\n[2/6] データセット構築")
print("-" * 100)

dataset = []
feature_names = None
max_index = len(df) - 24
sampling_interval = 4

print("処理中...")
for i in range(100, max_index, sampling_interval):
    if i % 2000 == 0:
        print(f"  進捗: {i}/{max_index} ({i/max_index*100:.1f}%)")

    features = calculate_advanced_features(df, i)
    if features is None:
        continue

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
print(f"特徴量数: {len(feature_names)} ← 16から大幅増加！")

PROFIT_THRESHOLD = 0.01
dataset_df['target'] = (dataset_df['net_return'] > PROFIT_THRESHOLD).astype(int)

print(f"\n買いシグナル: {dataset_df['target'].sum():,} ({dataset_df['target'].mean()*100:.1f}%)")

# ============================================================================
# 訓練・テスト分割
# ============================================================================
print("\n[3/6] 訓練・テスト分割")
print("-" * 100)

split_idx = int(len(dataset_df) * 0.8)
train_data = dataset_df.iloc[:split_idx]
test_data = dataset_df.iloc[split_idx:]

X_train = train_data[feature_names]
y_train = train_data['target']
X_test = test_data[feature_names]
y_test = test_data['target']

print(f"訓練: {len(X_train):,}, テスト: {len(X_test):,}")

# ============================================================================
# モデル訓練
# ============================================================================
print("\n[4/6] LightGBMモデル訓練")
print("-" * 100)

model = lgb.LGBMClassifier(
    n_estimators=150,  # 少し増やす
    max_depth=6,       # 少し深く
    learning_rate=0.05,  # 少し低く
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
test_buy = y_pred_test == 1
test_win_rate = y_test.values[test_buy].mean() if test_buy.sum() > 0 else 0

print(f"テスト買いシグナル勝率: {test_win_rate*100:.1f}%")
print(f"テスト買いシグナル数: {test_buy.sum()}")

# ============================================================================
# バックテスト
# ============================================================================
print("\n[5/6] バックテスト")
print("-" * 100)

backtest_data = test_data.copy()
backtest_data['prediction'] = y_pred_test

trades = []
current_equity = 1.0

for idx, row in backtest_data.iterrows():
    if row['prediction'] == 1:
        entry_idx = row['index']
        actual_return = calculate_realistic_return(df, entry_idx)
        if actual_return is not None:
            position_size = 0.02 * current_equity
            pnl = position_size * actual_return
            current_equity += pnl
            trades.append({
                'datetime': row['datetime'],
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

    returns_series = trades_df['return'].values
    mean_return = np.mean(returns_series)
    std_return = np.std(returns_series)
    sharpe_ratio = (mean_return / std_return) * np.sqrt(365 * 24) if std_return > 0 else 0

    equity_series = trades_df['equity'].values
    running_max = np.maximum.accumulate(equity_series)
    drawdowns = (equity_series - running_max) / running_max
    max_drawdown = np.min(drawdowns)

    print("\n" + "=" * 100)
    print("改善版V2 バックテスト結果")
    print("=" * 100)
    print(f"総トレード数: {n_trades}")
    print(f"勝率: {win_rate*100:.1f}%")
    print(f"平均勝ち: +{avg_win*100:.2f}%")
    print(f"平均負け: {avg_loss*100:.2f}%")
    print(f"Profit Factor: {abs(avg_win / avg_loss) if avg_loss != 0 else 0:.2f}")
    print(f"総リターン: {total_return*100:.2f}%")
    print(f"最大DD: {max_drawdown*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.3f}")

    # 比較
    print("\n" + "=" * 100)
    print("ベースライン vs 改善版V2")
    print("=" * 100)
    print(f"勝率:         46.3% → {win_rate*100:.1f}% ({(win_rate-0.463)*100:+.1f}%ポイント)")
    print(f"Sharpe:       1.646 → {sharpe_ratio:.3f} ({sharpe_ratio-1.646:+.3f})")
    print(f"Profit Factor: 1.20 → {abs(avg_win / avg_loss) if avg_loss != 0 else 0:.2f}")

    trades_df.to_csv('improved_v2_trades.csv', index=False)

# ============================================================================
# 特徴量重要度
# ============================================================================
print("\n[6/6] 特徴量重要度 TOP 15")
print("-" * 100)

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:.1f}")

feature_importance.to_csv('improved_v2_feature_importance.csv', index=False)

print("\n" + "=" * 100)
print("改善版V2完了！")
print("=" * 100)
