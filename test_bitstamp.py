import ccxt

try:
    print("Bitstampに接続中...")
    exchange = ccxt.bitstamp({'enableRateLimit': True})

    print("現在のBTC/USD価格を取得中...")
    ticker = exchange.fetch_ticker('BTC/USD')

    print(f"現在のBTC/USD価格: ${ticker['last']:,.2f}")
    print("接続成功!")

except Exception as e:
    print(f"エラー: {e}")
    import traceback
    traceback.print_exc()
