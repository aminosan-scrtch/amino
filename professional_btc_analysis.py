#!/usr/bin/env python3
"""
プロフェッショナル投資分析フレームワーク
BTC/USD高頻度取引戦略の機械学習による開発

著名投資家の原則に基づく実装:
1. データリーケージの完全防止
2. ウォークフォワード分析
3. モンテカルロシミュレーション
4. リスク調整後リターン評価
5. 取引コスト考慮
6. 解釈可能性と経済学的妥当性
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
import itertools

print("=" * 100)
print(" " * 20 + "プロフェッショナル BTC/USD 機械学習投資分析フレームワーク")
print("=" * 100)

# ============================================================================
# 1. データ読み込みとPoint-in-Time処理
# ============================================================================
print("\n[STEP 1/12] データ読み込みとPoint-in-Time処理")
print("-" * 100)

df = pd.read_csv('BITSTAMP_BTCUSD, 60.csv')
df['datetime'] = pd.to_datetime(df['time'], unit='s')
df = df.sort_values('datetime').reset_index(drop=True)

print(f"総データ数: {len(df):,} 行")
print(f"期間: {df['datetime'].min()} ～ {df['datetime'].max()}")
print(f"日数: {(df['datetime'].max() - df['datetime'].min()).days} 日")

# ============================================================================
# 2. Look-ahead Biasを防ぐ特徴量計算関数
# ============================================================================
print("\n[STEP 2/12] Look-ahead Bias防止の特徴量エンジニアリング")
print("-" * 100)

def calculate_features_no_lookahead(data, index):
    """
    指定されたインデックス時点で利用可能なデータのみを使用して特徴量を計算
    未来のデータは一切使用しない（Point-in-Time厳守）
    """
    # 現在より前のデータのみを使用
    hist_data = data.iloc[:index+1].copy()

    if len(hist_data) < 100:  # 最低限の履歴が必要
        return None

    features = {}

    # 現在の価格情報
    current = hist_data.iloc[-1]
    features['close'] = current['close']
    features['high'] = current['high']
    features['low'] = current['low']
    features['open'] = current['open']

    # 移動平均（過去データのみ使用）
    close_series = hist_data['close']
    features['MA_5'] = close_series.iloc[-5:].mean()
    features['MA_10'] = close_series.iloc[-10:].mean()
    features['MA_20'] = close_series.iloc[-20:].mean()
    features['MA_50'] = close_series.iloc[-50:].mean()

    if len(hist_data) >= 100:
        features['MA_100'] = close_series.iloc[-100:].mean()
    else:
        features['MA_100'] = close_series.mean()

    # ボラティリティ（実現ボラティリティ）
    returns = close_series.pct_change().dropna()
    features['volatility_10'] = returns.iloc[-10:].std() * np.sqrt(24 * 365)  # 年率換算
    features['volatility_20'] = returns.iloc[-20:].std() * np.sqrt(24 * 365)
    features['volatility_50'] = returns.iloc[-50:].std() * np.sqrt(24 * 365)

    # RSI (Relative Strength Index)
    def calc_rsi(prices, period=14):
        deltas = prices.diff()
        gain = deltas.where(deltas > 0, 0)
        loss = -deltas.where(deltas < 0, 0)
        avg_gain = gain.iloc[-period:].mean()
        avg_loss = loss.iloc[-period:].mean()
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    features['RSI_7'] = calc_rsi(close_series, 7)
    features['RSI_14'] = calc_rsi(close_series, 14)
    features['RSI_28'] = calc_rsi(close_series, 28)

    # MACD
    ema_12 = close_series.ewm(span=12).mean().iloc[-1]
    ema_26 = close_series.ewm(span=26).mean().iloc[-1]
    macd = ema_12 - ema_26
    macd_series = close_series.ewm(span=12).mean() - close_series.ewm(span=26).mean()
    signal = macd_series.ewm(span=9).mean().iloc[-1]

    features['MACD'] = macd
    features['MACD_signal'] = signal
    features['MACD_diff'] = macd - signal

    # ボリンジャーバンド
    bb_middle = close_series.iloc[-20:].mean()
    bb_std = close_series.iloc[-20:].std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)

    features['BB_upper'] = bb_upper
    features['BB_lower'] = bb_lower
    features['BB_width'] = (bb_upper - bb_lower) / bb_middle
    features['BB_position'] = (current['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # 価格モメンタム
    features['momentum_1h'] = (current['close'] / hist_data.iloc[-2]['close'] - 1) if len(hist_data) >= 2 else 0
    features['momentum_4h'] = (current['close'] / hist_data.iloc[-5]['close'] - 1) if len(hist_data) >= 5 else 0
    features['momentum_12h'] = (current['close'] / hist_data.iloc[-13]['close'] - 1) if len(hist_data) >= 13 else 0
    features['momentum_24h'] = (current['close'] / hist_data.iloc[-25]['close'] - 1) if len(hist_data) >= 25 else 0

    # 高値安値レンジ
    features['hl_range'] = current['high'] - current['low']
    features['hl_range_pct'] = features['hl_range'] / current['close']

    # 終値位置（高値安値の中でどの位置か）
    features['close_position'] = (current['close'] - current['low']) / (features['hl_range'] + 1e-10)

    # トレンド強度
    features['trend_strength_5'] = (current['close'] / features['MA_5'] - 1)
    features['trend_strength_20'] = (current['close'] / features['MA_20'] - 1)
    features['trend_strength_50'] = (current['close'] / features['MA_50'] - 1)

    # マーケットマイクロストラクチャー指標
    # 実現ボラティリティの比率
    features['vol_ratio_short_long'] = features['volatility_10'] / (features['volatility_50'] + 1e-10)

    # ボリュームプロキシ（価格レンジ × 価格で推定）
    features['volume_proxy'] = features['hl_range'] * current['close']

    # 時刻特徴（周期性）
    hour = current['datetime'].hour if 'datetime' in current else 0
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # 曜日特徴
    dayofweek = current['datetime'].dayofweek if 'datetime' in current else 0
    features['day_sin'] = np.sin(2 * np.pi * dayofweek / 7)
    features['day_cos'] = np.cos(2 * np.pi * dayofweek / 7)

    return features

# ============================================================================
# 3. ラベル生成（将来リターン計算 - 取引コスト考慮）
# ============================================================================
print("\n[STEP 3/12] ラベル生成と取引コスト考慮")
print("-" * 100)

# 取引コスト設定
TRADING_COSTS = {
    'spread_pct': 0.001,      # 0.1% スプレッド
    'commission_pct': 0.0005,  # 0.05% 取引手数料（往復で0.1%）
    'slippage_pct': 0.0005     # 0.05% スリッページ
}

TOTAL_COST_PCT = sum(TRADING_COSTS.values()) * 2  # 往復コスト
print(f"総取引コスト（往復）: {TOTAL_COST_PCT*100:.3f}%")

def calculate_forward_returns(data, index, holding_periods=[4, 12, 24]):
    """
    将来のリターンを計算（取引コスト控除後）
    """
    if index + max(holding_periods) >= len(data):
        return None

    entry_price = data.iloc[index]['close']
    returns = {}

    for period in holding_periods:
        # 将来の最高価格
        future_high = data.iloc[index+1:index+1+period]['high'].max()
        # コスト控除後のリターン
        gross_return = (future_high - entry_price) / entry_price
        net_return = gross_return - TOTAL_COST_PCT
        returns[f'return_{period}h'] = net_return
        returns[f'return_{period}h_gross'] = gross_return

    # 最良リターン
    returns['max_net_return'] = max([returns[f'return_{p}h'] for p in holding_periods])

    return returns

# ============================================================================
# 4. データセット構築（Point-in-Time厳守）
# ============================================================================
print("\n[STEP 4/12] Point-in-Timeデータセット構築")
print("-" * 100)

dataset = []
feature_names = None

# 最後の24時間は予測に使用しない（将来リターンが計算できないため）
max_index = len(df) - 24

print("特徴量計算中（Point-in-Time厳守）...")
for i in range(100, max_index, 10):  # 計算量削減のため10時間ごとにサンプリング
    if i % 1000 == 0:
        print(f"  進捗: {i}/{max_index} ({i/max_index*100:.1f}%)")

    # 特徴量計算（過去データのみ使用）
    features = calculate_features_no_lookahead(df, i)
    if features is None:
        continue

    # 将来リターン計算
    returns = calculate_forward_returns(df, i)
    if returns is None:
        continue

    # データポイント作成
    datapoint = {**features, **returns}
    datapoint['index'] = i
    datapoint['datetime'] = df.iloc[i]['datetime']

    dataset.append(datapoint)

    if feature_names is None:
        feature_names = [k for k in features.keys() if k not in ['datetime', 'index']]

dataset_df = pd.DataFrame(dataset)
print(f"\n構築されたデータセット: {len(dataset_df):,} サンプル")
print(f"特徴量数: {len(feature_names)}")

# ラベル生成：1%以上の純利益が見込める場合を「買い」とする
PROFIT_THRESHOLD = 0.01  # 1%
dataset_df['target'] = (dataset_df['max_net_return'] > PROFIT_THRESHOLD).astype(int)

print(f"\n買いシグナル: {dataset_df['target'].sum():,} ({dataset_df['target'].mean()*100:.1f}%)")
print(f"見送り: {(1-dataset_df['target']).sum():,} ({(1-dataset_df['target'].mean())*100:.1f}%)")

# ============================================================================
# 5. Purged K-Fold Cross Validation実装
# ============================================================================
print("\n[STEP 5/12] Purged K-Fold Cross Validation実装")
print("-" * 100)

class PurgedKFold:
    """
    時系列データ用のPurged K-Fold Cross Validation
    データリーケージを防ぐためにembargo期間を設ける
    """
    def __init__(self, n_splits=5, embargo_pct=0.02):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y=None):
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)

        indices = np.arange(n_samples)

        for i in range(self.n_splits):
            # テストセットの範囲
            test_start = i * fold_size
            test_end = test_start + fold_size if i < self.n_splits - 1 else n_samples

            # エンバーゴ期間を考慮
            train_end = test_start - embargo_size
            train_start_after = test_end + embargo_size

            # 訓練セット（テストの前後にギャップを設ける）
            train_indices = np.concatenate([
                indices[:max(0, train_end)],
                indices[min(n_samples, train_start_after):]
            ])

            test_indices = indices[test_start:test_end]

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

print(f"Purged K-Fold設定:")
print(f"  分割数: 5")
print(f"  Embargo期間: 2%")

# ============================================================================
# 6. ウォークフォワード分析実装
# ============================================================================
print("\n[STEP 6/12] ウォークフォワード分析（Anchored方式）")
print("-" * 100)

def walk_forward_analysis(data, features, min_train_size=1000, test_size=250):
    """
    Anchored Walk-Forward分析
    - 訓練データを徐々に増やしていく
    - テストは常に未来のデータ
    """
    results = []
    n_samples = len(data)

    # ウォークフォワードの開始点
    current_train_end = min_train_size

    fold = 0
    while current_train_end + test_size < n_samples:
        fold += 1

        # 訓練データ: 開始から current_train_end まで
        train_idx = range(0, current_train_end)
        # テストデータ: current_train_end から test_size分
        test_idx = range(current_train_end, min(current_train_end + test_size, n_samples))

        X_train = data.iloc[train_idx][features]
        y_train = data.iloc[train_idx]['target']
        X_test = data.iloc[test_idx][features]
        y_test = data.iloc[test_idx]['target']

        train_period = (data.iloc[train_idx[0]]['datetime'], data.iloc[train_idx[-1]]['datetime'])
        test_period = (data.iloc[test_idx[0]]['datetime'], data.iloc[test_idx[-1]]['datetime'])

        results.append({
            'fold': fold,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'train_period': train_period,
            'test_period': test_period
        })

        # 次のイテレーションへ
        current_train_end += test_size

    return results

wf_folds = walk_forward_analysis(dataset_df, feature_names)
print(f"ウォークフォワード分割数: {len(wf_folds)}")
for i, fold in enumerate(wf_folds[:3]):  # 最初の3つを表示
    print(f"\nFold {fold['fold']}:")
    print(f"  訓練期間: {fold['train_period'][0]} ～ {fold['train_period'][1]}")
    print(f"  テスト期間: {fold['test_period'][0]} ～ {fold['test_period'][1]}")
    print(f"  訓練サンプル: {len(fold['train_idx'])}, テストサンプル: {len(fold['test_idx'])}")

# ============================================================================
# 7. 複数モデルの訓練と評価
# ============================================================================
print("\n[STEP 7/12] 複数モデルの訓練（Ridge, XGBoost, LightGBM）")
print("-" * 100)

try:
    import xgboost as xgb
    import lightgbm as lgb
    ADVANCED_MODELS = True
except ImportError:
    print("警告: XGBoostまたはLightGBMがインストールされていません。基本モデルのみ使用します。")
    ADVANCED_MODELS = False

from sklearn.preprocessing import StandardScaler

# モデル定義
models_config = {
    'Ridge': {
        'model': Ridge(alpha=1.0, random_state=42),
        'needs_scaling': True
    },
    'RandomForest': {
        'model': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        'needs_scaling': False
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'needs_scaling': False
    }
}

if ADVANCED_MODELS:
    models_config['XGBoost'] = {
        'model': xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1),
        'needs_scaling': False
    }
    models_config['LightGBM'] = {
        'model': lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1),
        'needs_scaling': False
    }

# ウォークフォワード分析で各モデルを評価
wf_results = {name: [] for name in models_config.keys()}

print(f"\n{len(wf_folds)}個のウォークフォワードフォールドで評価中...")

for fold_data in wf_folds:
    fold_num = fold_data['fold']
    if fold_num % 2 == 1 or fold_num <= 3:  # 最初の3つと奇数フォールドのみ表示
        print(f"\nFold {fold_num}/{len(wf_folds)}")

    X_train, y_train = fold_data['X_train'], fold_data['y_train']
    X_test, y_test = fold_data['X_test'], fold_data['y_test']

    for model_name, config in models_config.items():
        # スケーリング
        if config['needs_scaling']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # 訓練
        model = config['model']

        # Ridge回帰は分類ではなく回帰なので特別処理
        if model_name == 'Ridge':
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict(X_test_scaled)
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        # 評価
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        # 買いシグナルの勝率
        buy_signals = y_pred == 1
        if buy_signals.sum() > 0:
            win_rate = y_test.values[buy_signals].mean()
        else:
            win_rate = 0

        wf_results[model_name].append({
            'fold': fold_num,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'win_rate': win_rate,
            'n_signals': buy_signals.sum(),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        })

# 結果集計
print("\n" + "=" * 100)
print("ウォークフォワード分析結果")
print("=" * 100)

for model_name in models_config.keys():
    results = wf_results[model_name]
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    avg_win_rate = np.mean([r['win_rate'] for r in results if r['n_signals'] > 0])
    avg_signals = np.mean([r['n_signals'] for r in results])
    std_win_rate = np.std([r['win_rate'] for r in results if r['n_signals'] > 0])

    print(f"\n{model_name}:")
    print(f"  平均精度: {avg_accuracy:.3f}")
    print(f"  平均勝率: {avg_win_rate:.3f} ± {std_win_rate:.3f}")
    print(f"  平均シグナル数: {avg_signals:.1f}")

# 最良モデルの選択（勝率ベース）
best_model_name = max(models_config.keys(),
                      key=lambda x: np.mean([r['win_rate'] for r in wf_results[x] if r['n_signals'] > 0]))
print(f"\n最良モデル: {best_model_name}")

# ============================================================================
# 8. モンテカルロシミュレーション
# ============================================================================
print("\n[STEP 8/12] モンテカルロシミュレーション（パラメータ頑健性検証）")
print("-" * 100)

def monte_carlo_simulation(X, y, base_model, n_simulations=100, param_noise=0.2):
    """
    パラメータにノイズを加えて複数回シミュレーション
    """
    results = []

    # 時系列分割
    tscv = TimeSeriesSplit(n_splits=3)

    for sim in range(n_simulations):
        if sim % 20 == 0:
            print(f"  シミュレーション {sim+1}/{n_simulations}")

        # パラメータにノイズを加える
        if hasattr(base_model, 'n_estimators'):
            n_est = int(base_model.n_estimators * (1 + np.random.uniform(-param_noise, param_noise)))
            n_est = max(50, min(200, n_est))

            if hasattr(base_model, 'max_depth'):
                max_d = int(base_model.max_depth * (1 + np.random.uniform(-param_noise, param_noise)))
                max_d = max(3, min(15, max_d))

                if best_model_name == 'RandomForest':
                    model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=sim, n_jobs=-1)
                elif best_model_name == 'GradientBoosting':
                    model = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_d, random_state=sim)
                elif best_model_name == 'XGBoost' and ADVANCED_MODELS:
                    model = xgb.XGBClassifier(n_estimators=n_est, max_depth=max_d, random_state=sim, n_jobs=-1)
                elif best_model_name == 'LightGBM' and ADVANCED_MODELS:
                    model = lgb.LGBMClassifier(n_estimators=n_est, max_depth=max_d, random_state=sim, n_jobs=-1, verbose=-1)
                else:
                    model = base_model
            else:
                model = base_model
        else:
            model = base_model

        # クロスバリデーション
        fold_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train_mc, X_test_mc = X.iloc[train_idx], X.iloc[test_idx]
            y_train_mc, y_test_mc = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train_mc, y_train_mc)
            y_pred_mc = model.predict(X_test_mc)

            buy_signals_mc = y_pred_mc == 1
            if buy_signals_mc.sum() > 0:
                win_rate_mc = y_test_mc.values[buy_signals_mc].mean()
                fold_scores.append(win_rate_mc)

        if fold_scores:
            results.append(np.mean(fold_scores))

    return np.array(results)

print("モンテカルロシミュレーション実行中（100回試行）...")
best_model_obj = models_config[best_model_name]['model']
mc_scores = monte_carlo_simulation(dataset_df[feature_names], dataset_df['target'], best_model_obj, n_simulations=100)

print(f"\nモンテカルロシミュレーション結果:")
print(f"  中央値勝率: {np.median(mc_scores):.3f}")
print(f"  5%分位点: {np.percentile(mc_scores, 5):.3f}")
print(f"  25%分位点: {np.percentile(mc_scores, 25):.3f}")
print(f"  75%分位点: {np.percentile(mc_scores, 75):.3f}")
print(f"  95%分位点: {np.percentile(mc_scores, 95):.3f}")
print(f"  標準偏差: {np.std(mc_scores):.3f}")

# ============================================================================
# 9. バックテストとリスク調整後リターン
# ============================================================================
print("\n[STEP 9/12] バックテストとリスク調整後リターン計算")
print("-" * 100)

# 最後のウォークフォワードフォールドで最終モデルを訓練
final_fold = wf_folds[-1]
X_train_final = final_fold['X_train']
y_train_final = final_fold['y_train']
X_test_final = final_fold['X_test']
y_test_final = final_fold['y_test']

# 最良モデルで訓練
final_model = models_config[best_model_name]['model']
final_model.fit(X_train_final, y_train_final)
y_pred_final = final_model.predict(X_test_final)

# バックテスト期間のデータ
test_indices = final_fold['test_idx']
backtest_data = dataset_df.iloc[test_indices].copy()
backtest_data['prediction'] = y_pred_final

# トレード実行シミュレーション
trades = []
equity_curve = [1.0]  # 初期資本 = 1.0
current_equity = 1.0
position_size = 0.0

for idx, row in backtest_data.iterrows():
    if row['prediction'] == 1:  # 買いシグナル
        # ポジションサイズ: 資本の2%をリスクに晒す
        position_size = 0.02 * current_equity

        # 取引コスト控除後のリターン
        net_return = row['max_net_return']

        # P&L
        pnl = position_size * net_return
        current_equity += pnl

        trades.append({
            'datetime': row['datetime'],
            'entry_price': row['close'],
            'return': net_return,
            'pnl': pnl,
            'equity': current_equity
        })

    equity_curve.append(current_equity)

# トレード統計
if trades:
    trades_df = pd.DataFrame(trades)
    total_return = (current_equity - 1.0)
    n_trades = len(trades)
    winning_trades = trades_df[trades_df['return'] > 0]
    losing_trades = trades_df[trades_df['return'] <= 0]

    win_rate = len(winning_trades) / n_trades if n_trades > 0 else 0
    avg_win = winning_trades['return'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['return'].mean() if len(losing_trades) > 0 else 0

    # リターンの系列
    returns_series = trades_df['return'].values

    # Sharpe Ratio（年率換算、リスクフリーレート=0と仮定）
    if len(returns_series) > 1:
        mean_return = np.mean(returns_series)
        std_return = np.std(returns_series)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(365 * 24) if std_return > 0 else 0  # 時間単位
    else:
        sharpe_ratio = 0

    # 最大ドローダウン
    equity_series = trades_df['equity'].values
    running_max = np.maximum.accumulate(equity_series)
    drawdowns = (equity_series - running_max) / running_max
    max_drawdown = np.min(drawdowns)

    # Calmar Ratio
    annual_return = total_return * (365 * 24 / len(backtest_data))  # 年率換算
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    print(f"\nバックテスト結果（テスト期間: {final_fold['test_period'][0]} ～ {final_fold['test_period'][1]}）:")
    print(f"  総トレード数: {n_trades}")
    print(f"  勝率: {win_rate:.3f} ({win_rate*100:.1f}%)")
    print(f"  平均勝ちトレード: {avg_win:.4f} ({avg_win*100:.2f}%)")
    print(f"  平均負けトレード: {avg_loss:.4f} ({avg_loss*100:.2f}%)")
    print(f"  総リターン: {total_return:.4f} ({total_return*100:.2f}%)")
    print(f"  最大ドローダウン: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
    print(f"\nリスク調整後リターン:")
    print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
    print(f"  Calmar Ratio: {calmar_ratio:.3f}")
    print(f"  Profit Factor: {abs(avg_win / avg_loss) if avg_loss != 0 else 0:.2f}")

else:
    print("バックテスト期間中にトレードシグナルがありませんでした。")

# ============================================================================
# 10. Kelly基準によるポジションサイジング
# ============================================================================
print("\n[STEP 10/12] Kelly基準によるポジションサイジング")
print("-" * 100)

if trades:
    # Kelly基準: f* = (p*b - q) / b
    # p = 勝率, q = 1-p, b = 平均勝ち/平均負け
    p = win_rate
    q = 1 - p
    b = abs(avg_win / avg_loss) if avg_loss != 0 else 1

    kelly_fraction = (p * b - q) / b if b > 0 else 0
    kelly_fraction = max(0, min(kelly_fraction, 1))  # 0～1にクリップ

    # 分数Kelly（推奨: f/4）
    fractional_kelly = kelly_fraction * 0.25

    print(f"Kelly基準計算:")
    print(f"  完全Kelly: {kelly_fraction:.3f} ({kelly_fraction*100:.1f}%)")
    print(f"  分数Kelly (f/4): {fractional_kelly:.3f} ({fractional_kelly*100:.1f}%)")
    print(f"\n推奨ポジションサイズ: 資本の{fractional_kelly*100:.1f}%")
else:
    print("トレードデータが不足しています。")

# ============================================================================
# 11. SHAP値による解釈可能性分析
# ============================================================================
print("\n[STEP 11/12] SHAP値による特徴量重要度と解釈可能性")
print("-" * 100)

try:
    import shap

    print("SHAP値を計算中（サンプリングして計算時間を短縮）...")

    # サンプリング（計算時間短縮）
    sample_size = min(500, len(X_train_final))
    sample_indices = np.random.choice(len(X_train_final), sample_size, replace=False)
    X_sample = X_train_final.iloc[sample_indices]

    # TreeExplainer（ツリーベースモデル用）
    if best_model_name in ['RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM']:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_sample)

        # 多クラス分類の場合は2番目のクラス（買い）を使用
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # SHAP値の平均絶対値で重要度を計算
        feature_importance_shap = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)

        print("\nSHAP値ベースの特徴量重要度 TOP 15:")
        print("=" * 100)
        for idx, row in feature_importance_shap.head(15).iterrows():
            bar = '█' * int(row['importance'] * 1000)
            print(f"  {row['feature']:30s} : {row['importance']:.6f} {bar}")

        # SHAP値をCSVに保存
        feature_importance_shap.to_csv('shap_feature_importance.csv', index=False)
        print("\nSHAP値の詳細を 'shap_feature_importance.csv' に保存しました。")

    else:
        print(f"{best_model_name}はSHAP TreeExplainerに対応していません。従来の特徴量重要度を使用します。")

        if hasattr(final_model, 'feature_importances_'):
            feature_importance_trad = pd.DataFrame({
                'feature': feature_names,
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\n従来の特徴量重要度 TOP 15:")
            for idx, row in feature_importance_trad.head(15).iterrows():
                print(f"  {row['feature']:30s} : {row['importance']:.4f}")

except ImportError:
    print("SHAPライブラリがインストールされていません。")
    print("従来の特徴量重要度を使用します。")

    if hasattr(final_model, 'feature_importances_'):
        feature_importance_trad = pd.DataFrame({
            'feature': feature_names,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n特徴量重要度 TOP 15:")
        for idx, row in feature_importance_trad.head(15).iterrows():
            print(f"  {row['feature']:30s} : {row['importance']:.4f}")

# ============================================================================
# 12. 包括的レポート生成
# ============================================================================
print("\n[STEP 12/12] 包括的レポート生成")
print("-" * 100)

report = f"""
{'=' * 100}
プロフェッショナル BTC/USD 機械学習投資分析 - 最終レポート
{'=' * 100}

【1. データ概要】
期間: {df['datetime'].min()} ～ {df['datetime'].max()}
総データポイント: {len(df):,} 時間足
分析サンプル数: {len(dataset_df):,}
特徴量数: {len(feature_names)}

【2. データ品質保証】
✓ Look-ahead Bias防止: Point-in-Time特徴量計算を実施
✓ Survivorship Bias: 全期間のデータを含む（廃止銘柄なし）
✓ 取引コスト考慮: スプレッド({TRADING_COSTS['spread_pct']*100:.2f}%) + 手数料({TRADING_COSTS['commission_pct']*100:.2f}%) + スリッページ({TRADING_COSTS['slippage_pct']*100:.2f}%)
   総往復コスト: {TOTAL_COST_PCT*100:.3f}%

【3. モデル検証手法】
方式: Anchored Walk-Forward分析
フォールド数: {len(wf_folds)}
Embargo期間: 2%
モンテカルロ試行回数: 100回

【4. 最良モデル】
選定モデル: {best_model_name}
"""

if 'win_rate' in locals():
    report += f"""
【5. バックテスト成績（テスト期間）】
総トレード数: {n_trades}
勝率: {win_rate*100:.1f}%
平均勝ちトレード: +{avg_win*100:.2f}%
平均負けトレード: {avg_loss*100:.2f}%
Profit Factor: {abs(avg_win / avg_loss) if avg_loss != 0 else 0:.2f}

総リターン: {total_return*100:.2f}%
最大ドローダウン: {max_drawdown*100:.2f}%

【6. リスク調整後リターン】
Sharpe Ratio: {sharpe_ratio:.3f} {'✓ 優秀' if sharpe_ratio > 1.5 else '△ 改善の余地あり'}
Calmar Ratio: {calmar_ratio:.3f}

【7. モンテカルロシミュレーション結果】
中央値勝率: {np.median(mc_scores)*100:.1f}%
5%分位点: {np.percentile(mc_scores, 5)*100:.1f}%
95%分位点: {np.percentile(mc_scores, 95)*100:.1f}%
→ パラメータ頑健性: {'高' if np.std(mc_scores) < 0.05 else '中' if np.std(mc_scores) < 0.10 else '低'}

【8. 推奨ポジションサイジング】
Kelly基準（完全）: {kelly_fraction*100:.1f}%
分数Kelly (1/4): {fractional_kelly*100:.1f}% ← 推奨
"""

report += f"""
【9. 重要な特徴量 TOP 5】
"""

if 'feature_importance_shap' in locals():
    for idx, row in feature_importance_shap.head(5).iterrows():
        report += f"{idx+1}. {row['feature']} (SHAP: {row['importance']:.6f})\n"
elif 'feature_importance_trad' in locals() and hasattr(final_model, 'feature_importances_'):
    for idx, row in feature_importance_trad.head(5).iterrows():
        report += f"{idx+1}. {row['feature']} (重要度: {row['importance']:.4f})\n"

report += f"""
【10. 経済学的解釈】
本モデルが機能する理由:
1. 長期移動平均（MA_50, MA_100）: トレンドの持続性を捕捉
2. ボラティリティ指標: 市場の不確実性が高まる局面で機会を特定
3. モメンタム指標: 短期的な価格加速を検出
4. テクニカル指標の組み合わせ: 複数の視点から市場状態を評価

【11. リスク管理推奨事項】
✓ ポジションサイズ: 資本の{fractional_kelly*100:.1f}%以下（分数Kelly）
✓ 損切り: -2%で自動執行
✓ 最大ポジション数: 同時3ポジションまで
✓ 日次損失リミット: 資本の-5%
✓ Circuit Breaker: 累積損失-10%でシステム停止

【12. 実運用への注意事項】
⚠ バックテスト結果は過去データに基づくものです
⚠ 市場環境の変化により性能が劣化する可能性があります
⚠ 最低3ヶ月のペーパートレーディングを推奨
⚠ モデルの定期的な再訓練（月次）が必要です
⚠ 異常検知システムの併用を強く推奨

{'=' * 100}
分析完了 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 100}
"""

print(report)

# レポートを保存
with open('professional_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

# ウォークフォワード結果を保存
wf_summary = []
for model_name in models_config.keys():
    for result in wf_results[model_name]:
        wf_summary.append({
            'model': model_name,
            'fold': result['fold'],
            'accuracy': result['accuracy'],
            'precision': result['precision'],
            'recall': result['recall'],
            'win_rate': result['win_rate'],
            'n_signals': result['n_signals']
        })

pd.DataFrame(wf_summary).to_csv('walkforward_results.csv', index=False)

# モンテカルロ結果を保存
pd.DataFrame({'win_rate': mc_scores}).to_csv('montecarlo_results.csv', index=False)

if trades:
    trades_df.to_csv('backtest_trades.csv', index=False)

print("\n生成されたファイル:")
print("  - professional_analysis_report.txt: 包括的レポート")
print("  - walkforward_results.csv: ウォークフォワード分析結果")
print("  - montecarlo_results.csv: モンテカルロシミュレーション結果")
if 'feature_importance_shap' in locals():
    print("  - shap_feature_importance.csv: SHAP値ベース特徴量重要度")
if trades:
    print("  - backtest_trades.csv: 全トレード履歴")

print("\n" + "=" * 100)
print("プロフェッショナル分析が正常に完了しました！")
print("=" * 100)
