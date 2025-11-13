"""
Bitstamp BTC/USD データ取得スクリプト（設定変更可能版）

使い方:
1. pip install ccxt pandas
2. 下記の設定変数を変更
3. python get_bitstamp_data.py

出力: bitstamp_btcusd_YYYYMMDD_HHMM_Xm.csv と .json
"""

import ccxt
import pandas as pd
from datetime import datetime
import time
import json
import pytz

# ========================================
# 設定変数（ここを変更してください）
# ========================================

# 日本時間での取得期間
START_TIME_JST = '2025-10-21 13:00:00'  # 開始時刻（日本時間）
END_TIME_JST = '2025-10-21 15:30:00'    # 終了時刻（日本時間）

# 時間足（'1m', '5m', '15m', '1h' など）
TIMEFRAME = '1m'

# ========================================


def jst_to_utc(jst_time_str):
    """
    日本時間の文字列をUTC時刻の文字列に変換

    Args:
        jst_time_str: 日本時間の文字列 (例: '2025-10-21 13:00:00')

    Returns:
        UTC時刻の文字列 (例: '2025-10-21T04:00:00Z')
    """
    jst = pytz.timezone('Asia/Tokyo')
    utc = pytz.utc

    # 日本時間でdatetimeオブジェクトを作成
    jst_dt = jst.localize(datetime.strptime(jst_time_str, '%Y-%m-%d %H:%M:%S'))

    # UTCに変換
    utc_dt = jst_dt.astimezone(utc)

    # ISO形式の文字列で返す
    return utc_dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def generate_filename(start_time_jst, timeframe):
    """
    ファイル名を生成（開始日時と時間足を含む）

    Args:
        start_time_jst: 開始時刻（日本時間の文字列）
        timeframe: 時間足 (例: '1m', '5m')

    Returns:
        ファイル名のベース (例: 'bitstamp_btcusd_20251021_1300_1m')
    """
    dt = datetime.strptime(start_time_jst, '%Y-%m-%d %H:%M:%S')
    date_str = dt.strftime('%Y%m%d_%H%M')
    return f'bitstamp_btcusd_{date_str}_{timeframe}'


def fetch_ohlcv_paginated(exchange, symbol, timeframe, start_date_utc, end_date_utc):
    """
    ページネーションでデータを取得
    """
    all_data = []

    since = exchange.parse8601(start_date_utc)
    end_ms = exchange.parse8601(end_date_utc)

    print(f"データ取得開始: {start_date_utc} から {end_date_utc} (UTC)")
    print(f"取得中...", end="", flush=True)

    request_count = 0

    # 時間足からミリ秒を計算
    timeframe_ms = {
        '1m': 60000,
        '5m': 300000,
        '15m': 900000,
        '30m': 1800000,
        '1h': 3600000,
        '4h': 14400000,
        '1d': 86400000
    }
    interval_ms = timeframe_ms.get(timeframe, 60000)

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

            # 次の開始時刻を設定（最後のタイムスタンプ + 時間足）
            since = ohlcv[-1][0] + interval_ms

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
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

    # 終了時刻でフィルタ（タイムゾーン付きに変換）
    end_dt = pd.to_datetime(end_date_utc, utc=True)
    df = df[df['timestamp'] <= end_dt]

    # 重複削除
    df = df.drop_duplicates(subset=['timestamp'])

    # 日本時間の列を追加（タイムゾーンなしに変換）
    df['timestamp_jst'] = df['timestamp'].dt.tz_convert('Asia/Tokyo').dt.tz_localize(None)

    return df


def main():
    print("=" * 60)
    print("Bitstamp BTC/USD データ取得")
    print("=" * 60)
    print()

    # 設定情報の表示
    print("設定情報:")
    print(f"  開始時刻（日本時間）: {START_TIME_JST}")
    print(f"  終了時刻（日本時間）: {END_TIME_JST}")
    print(f"  時間足: {TIMEFRAME}")
    print()

    # 取引所接続
    print("Bitstampに接続中...")
    exchange = ccxt.bitstamp({
        'enableRateLimit': True,  # レート制限を自動で守る
    })

    # 日本時間をUTCに変換
    start_utc = jst_to_utc(START_TIME_JST)
    end_utc = jst_to_utc(END_TIME_JST)

    print(f"UTC変換結果: {start_utc} ～ {end_utc}")
    print()

    # データ取得
    df = fetch_ohlcv_paginated(
        exchange,
        'BTC/USD',
        TIMEFRAME,
        start_utc,
        end_utc
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

    # ファイル名を生成
    filename_base = generate_filename(START_TIME_JST, TIMEFRAME)

    # CSV保存
    df_output = df[['timestamp_jst', 'open', 'high', 'low', 'close', 'volume']].copy()
    df_output.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    csv_file = f'{filename_base}.csv'
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

    json_file = f'{filename_base}.json'
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
