"""
Bitstamp BTC/USD 1分足データ取得スクリプト

使い方:
1. pip install ccxt pandas
2. python get_bitstamp_data.py

取得期間: 2025-10-21 13:00:00 ～ 15:30:00 (日本時間)
出力: bitstamp_btcusd_1m.csv と bitstamp_btcusd_1m.json
"""

import ccxt
import pandas as pd
from datetime import datetime
import time
import json

def fetch_ohlcv_paginated(exchange, symbol, timeframe, start_date_utc, end_date_utc):
    """
    ページネーションで1分足データを取得
    """
    all_data = []

    since = exchange.parse8601(start_date_utc)
    end_ms = exchange.parse8601(end_date_utc)

    print(f"データ取得開始: {start_date_utc} から {end_date_utc} (UTC)")
    print(f"取得中...", end="", flush=True)

    request_count = 0

    while since < end_ms:
        try:
            # データ取得（最大1000本）
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)

            if not ohlcv:
                break

            # データ追加
            all_data.extend(ohlcv)
            request_count += 1

            # 進捗表示
            print(".", end="", flush=True)

            # 次の開始時刻を設定（最後のタイムスタンプ + 1分）
            since = ohlcv[-1][0] + 60000  # 1分 = 60000ミリ秒

            # API制限を考慮して待機
            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:
            print(f"\nエラー: {e}")
            print("5秒待機してリトライ...")
            time.sleep(5)
            continue

    print(f"\n{request_count}回のAPIリクエストを実行")

    # DataFrameに変換
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 終了時刻でフィルタ
    end_dt = pd.to_datetime(end_date_utc)
    df = df[df['timestamp'] <= end_dt]

    # 重複削除
    df = df.drop_duplicates(subset=['timestamp'])

    # 日本時間の列を追加
    df['timestamp_jst'] = df['timestamp'] + pd.Timedelta(hours=9)

    return df


def main():
    print("=" * 60)
    print("Bitstamp BTC/USD 1分足データ取得")
    print("=" * 60)
    print()

    # 取引所接続
    print("Bitstampに接続中...")
    exchange = ccxt.bitstamp({
        'enableRateLimit': True,  # レート制限を自動で守る
    })

    # データ取得
    # 日本時間 2025-10-21 13:00:00 = UTC 2025-10-21 04:00:00
    # 日本時間 2025-10-21 15:30:00 = UTC 2025-10-21 06:30:00
    df = fetch_ohlcv_paginated(
        exchange,
        'BTC/USD',
        '1m',
        '2025-10-21T04:00:00Z',  # UTC
        '2025-10-21T06:30:00Z'   # UTC
    )

    if len(df) == 0:
        print("\n警告: データが取得できませんでした")
        return

    print(f"\n合計 {len(df)} 本のデータを取得しました")
    print()

    # データの確認
    print("=" * 60)
    print("データサマリー")
    print("=" * 60)
    print(f"開始時刻（日本時間）: {df['timestamp_jst'].iloc[0]}")
    print(f"終了時刻（日本時間）: {df['timestamp_jst'].iloc[-1]}")
    print(f"始値: ${df['open'].iloc[0]:,.2f}")
    print(f"終値: ${df['close'].iloc[-1]:,.2f}")
    print(f"最高値: ${df['high'].max():,.2f}")
    print(f"最安値: ${df['low'].min():,.2f}")
    print(f"変動: ${df['close'].iloc[-1] - df['open'].iloc[0]:+,.2f}")
    print()

    print("最初の5行:")
    print(df[['timestamp_jst', 'open', 'high', 'low', 'close', 'volume']].head())
    print()
    print("最後の5行:")
    print(df[['timestamp_jst', 'open', 'high', 'low', 'close', 'volume']].tail())
    print()

    # CSV保存
    df_output = df[['timestamp_jst', 'open', 'high', 'low', 'close', 'volume']].copy()
    df_output.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    csv_file = 'bitstamp_btcusd_1m.csv'
    df_output.to_csv(csv_file, index=False)
    print(f"✓ CSVファイル保存: {csv_file}")

    # JSON保存
    json_data = []
    for _, row in df_output.iterrows():
        json_data.append({
            'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })

    json_file = 'bitstamp_btcusd_1m.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"✓ JSONファイル保存: {json_file}")

    print()
    print("=" * 60)
    print("完了！")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中断されました")
    except Exception as e:
        print(f"\n\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
