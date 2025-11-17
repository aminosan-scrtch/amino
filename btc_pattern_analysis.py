#!/usr/bin/env python3
"""
BTC/USD Pattern Analysis using Machine Learning
最も勝率が高いトレーディングパターンを見つける
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("BTC/USD 機械学習パターン分析")
print("=" * 80)

# データの読み込み
print("\n[1/6] データの読み込み中...")
df = pd.read_csv('BITSTAMP_BTCUSD, 60.csv')
print(f"データ行数: {len(df):,}")
print(f"期間: {pd.to_datetime(df['time'].iloc[0], unit='s')} から {pd.to_datetime(df['time'].iloc[-1], unit='s')}")

# テクニカル指標の計算
print("\n[2/6] テクニカル指標の計算中...")

# 移動平均
df['MA_5'] = df['close'].rolling(window=5).mean()
df['MA_10'] = df['close'].rolling(window=10).mean()
df['MA_20'] = df['close'].rolling(window=20).mean()
df['MA_50'] = df['close'].rolling(window=50).mean()
df['MA_100'] = df['close'].rolling(window=100).mean()

# ボラティリティ
df['volatility_10'] = df['close'].rolling(window=10).std()
df['volatility_20'] = df['close'].rolling(window=20).std()

# RSI (Relative Strength Index)
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI_14'] = calculate_rsi(df['close'], 14)
df['RSI_7'] = calculate_rsi(df['close'], 7)

# MACD
def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

df['MACD'], df['MACD_signal'] = calculate_macd(df['close'])
df['MACD_diff'] = df['MACD'] - df['MACD_signal']

# ボリンジャーバンド
df['BB_middle'] = df['close'].rolling(window=20).mean()
df['BB_std'] = df['close'].rolling(window=20).std()
df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

# 価格変化率
df['price_change_1h'] = df['close'].pct_change(1)
df['price_change_4h'] = df['close'].pct_change(4)
df['price_change_24h'] = df['close'].pct_change(24)

# 高値・安値との関係
df['high_low_range'] = df['high'] - df['low']
df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

# ボリューム的な特徴（価格レンジから推測）
df['volume_proxy'] = df['high_low_range'] * df['close']

# トレンド指標
df['trend_5'] = df['close'] / df['MA_5'] - 1
df['trend_20'] = df['close'] / df['MA_20'] - 1
df['trend_50'] = df['close'] / df['MA_50'] - 1

print(f"計算したテクニカル指標数: {len(df.columns) - 5}")

# ラベル付け：将来の価格変動を予測
print("\n[3/6] トレーディングシグナルの生成中...")

# 複数の時間軸で利益が出るパターンを探す
future_periods = [1, 4, 12, 24]  # 1時間後、4時間後、12時間後、24時間後

# 将来の最高リターンを計算
profit_threshold = 0.01  # 1%以上の利益を「勝ち」とする

df['future_max_return'] = 0
for period in future_periods:
    future_max = df['high'].shift(-period).rolling(window=period).max()
    returns = (future_max - df['close']) / df['close']
    df['future_max_return'] = df['future_max_return'].combine(returns, max)

# ラベル：1%以上の利益が見込める場合は1（買い）、それ以外は0
df['target'] = (df['future_max_return'] > profit_threshold).astype(int)

print(f"買いシグナル（勝ち）: {df['target'].sum():,} ({df['target'].mean()*100:.1f}%)")
print(f"見送りシグナル: {(1-df['target']).sum():,} ({(1-df['target'].mean())*100:.1f}%)")

# 欠損値を除去
df_clean = df.dropna()
print(f"\n前処理後のデータ行数: {len(df_clean):,}")

# 特徴量とターゲットの準備
feature_columns = [
    'MA_5', 'MA_10', 'MA_20', 'MA_50', 'MA_100',
    'volatility_10', 'volatility_20',
    'RSI_14', 'RSI_7',
    'MACD', 'MACD_signal', 'MACD_diff',
    'BB_position',
    'price_change_1h', 'price_change_4h', 'price_change_24h',
    'high_low_range', 'close_position',
    'trend_5', 'trend_20', 'trend_50'
]

X = df_clean[feature_columns]
y = df_clean['target']

# 訓練データとテストデータに分割（時系列を考慮）
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\n訓練データ: {len(X_train):,}")
print(f"テストデータ: {len(X_test):,}")

# 機械学習モデルの訓練
print("\n[4/6] 機械学習モデルの訓練中...")

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n{name} を訓練中...")
    model.fit(X_train, y_train)

    # 予測
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 精度
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    # 買いシグナルの勝率を計算
    buy_signals = y_pred_test == 1
    if buy_signals.sum() > 0:
        win_rate = y_test[buy_signals].mean()
    else:
        win_rate = 0

    results[name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'win_rate': win_rate,
        'predictions': y_pred_test
    }

    print(f"  訓練精度: {train_acc:.3f}")
    print(f"  テスト精度: {test_acc:.3f}")
    print(f"  買いシグナル数: {buy_signals.sum()}")
    print(f"  買いシグナルの勝率: {win_rate:.3f} ({win_rate*100:.1f}%)")

# 最良のモデルを選択
best_model_name = max(results, key=lambda x: results[x]['win_rate'])
best_model = results[best_model_name]['model']

print(f"\n最良モデル: {best_model_name}")
print(f"勝率: {results[best_model_name]['win_rate']*100:.1f}%")

# 最も重要な特徴量（パターン）を分析
print("\n[5/6] 最も勝率が高いパターンを特定中...")

# 特徴量の重要度
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*80)
print("【最も重要なトレーディングパターン TOP 10】")
print("="*80)
for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']:25s} : {row['importance']:.4f} {'█' * int(row['importance'] * 100)}")

# パターンの詳細分析
print("\n" + "="*80)
print("【高勝率パターンの詳細分析】")
print("="*80)

# 買いシグナルを出したケースの分析
buy_indices = results[best_model_name]['predictions'] == 1
if buy_indices.sum() > 0:
    buy_data = X_test[buy_indices]

    print(f"\n買いシグナル総数: {buy_indices.sum()}")
    print(f"実際の勝率: {y_test[buy_indices].mean()*100:.1f}%")

    print("\n買いシグナル時の平均的なパターン:")
    for feature in feature_importance.head(10)['feature']:
        mean_val = buy_data[feature].mean()
        overall_mean = X_train[feature].mean()
        print(f"  {feature:25s}: {mean_val:10.4f} (全体平均: {overall_mean:10.4f})")

# 最終レポート
print("\n" + "="*80)
print("【最終レポート】")
print("="*80)

print(f"""
分析対象: BTC/USD 60分足
データ期間: {pd.to_datetime(df['time'].iloc[0], unit='s').strftime('%Y-%m-%d')} ～ {pd.to_datetime(df['time'].iloc[-1], unit='s').strftime('%Y-%m-%d')}
総データ数: {len(df):,} 行

使用モデル: {best_model_name}
テスト期間の勝率: {results[best_model_name]['win_rate']*100:.1f}%
買いシグナル数: {buy_indices.sum()}
利益目標: {profit_threshold*100}% 以上

最も重要な3つのパターン:
1. {feature_importance.iloc[0]['feature']} (重要度: {feature_importance.iloc[0]['importance']:.4f})
2. {feature_importance.iloc[1]['feature']} (重要度: {feature_importance.iloc[1]['importance']:.4f})
3. {feature_importance.iloc[2]['feature']} (重要度: {feature_importance.iloc[2]['importance']:.4f})

推奨戦略:
- これらのパターンが揃った時にのみエントリー
- リスク管理として損切りラインを設定（例：-2%）
- バックテストと実運用では結果が異なる可能性があることに注意
""")

print("="*80)
print("分析完了！")
print("="*80)

# 結果を保存
results_df = pd.DataFrame({
    'model': list(results.keys()),
    'train_accuracy': [results[k]['train_acc'] for k in results],
    'test_accuracy': [results[k]['test_acc'] for k in results],
    'win_rate': [results[k]['win_rate'] for k in results]
})

results_df.to_csv('ml_results.csv', index=False)
feature_importance.to_csv('feature_importance.csv', index=False)

print("\n結果ファイル:")
print("  - ml_results.csv: モデル比較結果")
print("  - feature_importance.csv: 特徴量の重要度")
