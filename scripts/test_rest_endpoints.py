"""Quick test of exchange REST endpoints for SOL, ADA, XRP."""
import json
import urllib.request

headers = {"User-Agent": "Mozilla/5.0"}

print("--- Testing Binance ---")
for sym in ["SOLUSDT", "ADAUSDT", "XRPUSDT"]:
    try:
        url = f"https://api.binance.com/api/v3/ticker/bookTicker?symbol={sym}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=4) as resp:
            d = json.loads(resp.read().decode())
            print(f"Binance {sym}: bid={d['bidPrice']}, ask={d['askPrice']}")
    except Exception as e:
        print(f"Binance {sym} ERR: {e}")

print("\n--- Testing Coinbase USD & USDT ---")
for prod in ["SOL-USD", "SOL-USDT", "ADA-USD", "ADA-USDT", "XRP-USD", "XRP-USDT"]:
    try:
        url = f"https://api.exchange.coinbase.com/products/{prod}/ticker"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=4) as resp:
            d = json.loads(resp.read().decode())
            print(f"Coinbase {prod}: bid={d.get('bid')}, ask={d.get('ask')}, vol_24h={d.get('volume')}")
    except Exception as e:
        print(f"Coinbase {prod} ERR: {e}")

print("\n--- Testing Kraken ---")
for pair in ["SOLUSDT", "ADAUSDT", "XRPUSDT", "SOLUSD", "ADAUSD", "XRPUSD"]:
    try:
        url = f"https://api.kraken.com/0/public/Ticker?pair={pair}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=4) as resp:
            d = json.loads(resp.read().decode())
            res = d.get("result", {})
            k = list(res.keys())[0] if res else None
            if k:
                print(f"Kraken {pair} ({k}): bid={res[k]['b'][0]}, ask={res[k]['a'][0]}")
    except Exception as e:
        print(f"Kraken {pair} ERR: {e}")
