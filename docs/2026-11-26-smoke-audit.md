# ArbSense collector audit · 26 November 2026

Two sequential 600-second public-feed runs used separate, initially absent SQLite databases on the same laptop. Both included BTC, ETH and SOL. The baseline used v2.2 with every tick archived at depth. The verification used v2.3 from clean commit `2f74a2e`. Each collector shut down after its timed run. The historical project database was outside both runs.

## Storage

The baseline and verification commands were `python scripts/run_live.py 600 baseline600.db` and `python scripts/run_live.py 600 smoke600.db`, with the output databases in a separate working directory. Rates below use the requested 600-second interval and decimal GB. The supplemental output also shows the actual span from first to last tick. The baseline kept processing some messages during shutdown.

```text
raw_rows=243294 seconds=600 rows/s=405.49 DB_bytes=213495808 MB/min=21.350 gross_GB/day=30.743
permanent_upper_GB/day=0.063 projected_5day_DB_GB=153.717 replay_hours=120
```

`gross_GB/day` extrapolates incoming writes. The five-day estimate includes six hours of raw tick pages and five days of permanent data at the measured rate. The baseline's 120-hour figure represents the full five days of unbounded accumulation. The permanent daily estimate uses a compacted private copy after dropping raw ticks and includes schema pages. File growth can vary with market traffic, event frequency and SQLite page reuse. Budget separate space for existing databases, WAL and backups.

## Verification output

`python scripts/audit_capture.py smoke600.db --seconds 600`

```text
raw_rows=290928 seconds=600 rows/s=484.88 DB_bytes=90513408 MB/min=9.051 gross_GB/day=13.034
permanent_upper_GB/day=0.231 projected_5day_DB_GB=4.357 replay_hours=6
symbol buy sell priced missing_books insufficient_depth conversion_required capture_ended
BTC/USDT binance coinbase 0 0 0 2 0
BTC/USDT binance kraken 3 1 0 0 0
BTC/USDT coinbase binance 0 0 0 16 0
BTC/USDT coinbase kraken 0 0 0 5 0
BTC/USDT kraken binance 2 0 0 0 0
BTC/USDT kraken coinbase 0 0 0 0 0
ETH/USDT binance coinbase 0 0 0 2 0
ETH/USDT binance kraken 3 4 0 0 0
ETH/USDT coinbase binance 0 0 0 1 0
ETH/USDT coinbase kraken 0 0 0 5 0
ETH/USDT kraken binance 0 0 0 0 0
ETH/USDT kraken coinbase 0 0 0 0 0
SOL/USDT binance kraken 0 0 0 0 0
SOL/USDT kraken binance 0 0 0 0 0
fee n min_USDT p10 median p90 max mean positive_share boot95_low boot95_high
retail 8 -15.13660 -15.05582 -14.50772 -4.07573 -4.07410 -10.76483 0.00000 0.00000 0.00000
vip1 8 -9.95333 -9.87058 -9.32748 -2.63163 -2.62936 -6.98376 0.00000 0.00000 0.00000
vip3 8 -6.59948 -6.51547 -5.97539 -1.69725 -1.69458 -4.53718 0.00000 0.00000 0.00000
zero_maker 8 -8.27641 -8.19302 -7.65143 -2.16444 -2.16197 -5.76047 0.00000 0.00000 0.00000
priced_minute_clusters 5
exchange n p50_ms p90_ms p99_ms max_ms
binance 0 n/a
coinbase 20750 142.55 233.23 834.76 1899.83
kraken 254062 149.46 401.33 1179.17 18541.08
clock 2026-11-26T05:22:38.272399+00:00 9.891986846923828 3.2106516882777214 NTP:in.pool.ntp.org 67.84372916445136
clock 2026-11-26T05:27:53.407502+00:00 21.35145664215088 28.33442483097315 NTP:pool.ntp.org 55.533789563924074
clock 2026-11-26T05:32:38.970841+00:00 17.965316772460938 3.002215176820755 NTP:pool.ntp.org 67.93787516653538
```

The smoke capture contained 44 events, including 8 with two priced legs and 31 requiring currency conversion. Returns use the configured quantity and taker fee on each side. The four scenario rows reprice the same events. The 95% percentile interval resamples one-minute clusters containing priced observations with replacement, using 10,000 draws and seed 42. When every observed outcome has the same sign, the empirical interval collapses to one value. A short capture provides little information about rare wins.

## Independent checks on the captured data

The retention check ran against an in-memory copy of the smoke database with the reference time advanced seven hours. The SHA-256 comparison covers event IDs, feature windows, lifecycle ticks and arrival evidence in event-ID order. The WAL and shared-memory high-water figure below is a sampled observation, with five-second intervals beginning after startup.

```text
sqlite_version 3.53.1
metadata ('v2.3', '2f74a2e024a6826fdd33ff20e7ab4ec8c0bb60a5', 0, 85.0)
baseline first_to_last_tick_s=608.349 rows/s=399.925 MB/min=21.057 GB/day=30.321
smoke first_to_last_tick_s=599.812 rows/s=485.032 MB/min=9.054 GB/day=13.038
raw_rows, with_depth (290928, 36093)
Coinbase exchange-timestamped receipt ages over 500 ms
('BTC/USD', 9642, 259, 113.93, 1883.73)
('ETH/USD', 9208, 261, 113.67, 1857.8)
('USDT/USD', 1900, 78, 114.36, 1899.83)
seconds_since_first_coinbase_tick, slow_count, n
0 385 2147
60 6 1987
120 134 2051
180 0 2164
240 10 2044
300 2 1981
360 2 1994
420 47 2118
480 10 2148
540 2 2116
Raw Coinbase samples: receive_epoch_ms, age_ms, product_id, raw_exchange_time, change_count
1795670561165.091 1327.208 ETH-USD 2026-11-26T05:22:39.847768Z 14
1795670561172.713 1314.861 BTC-USD 2026-11-26T05:22:39.867737Z 7
1795670561183.083 1289.218 ETH-USD 2026-11-26T05:22:39.903750Z 19
1795670561184.195 1275.300 BTC-USD 2026-11-26T05:22:39.918780Z 5
1795670561185.134 1227.111 ETH-USD 2026-11-26T05:22:39.967908Z 73
1795670561186.510 1202.887 BTC-USD 2026-11-26T05:22:39.993508Z 24
1795670561187.357 1172.465 ETH-USD 2026-11-26T05:22:40.024777Z 37
1795670561188.571 1143.339 BTC-USD 2026-11-26T05:22:40.055117Z 9
max_sampled_WAL_plus_SHM_bytes 4301120
private_copy_7h_later raw_pruned raw_remaining events_before events_after permanent_SHA256_equal
290928 0 44 44 True
```

## Coinbase delays

Across the eight raw samples above, exchange timestamps advanced about 207 ms while receipt times advanced about 23 ms. The initial NTP error bound was about 68 ms. The burst shows queued updates arriving together and then catching up. Each raw payload below groups multiple book changes. Coinbase documents a 50 ms batch period and identifies the payload time as the most recent message in that batch. The measured delay includes any server, network, WebSocket and local scheduling queues. These observations identify a backlog in the sampled burst, but cannot locate that queue or establish the cause of every outlier in the run. [Coinbase Exchange WebSocket channels](https://docs.cdp.coinbase.com/exchange/websocket-feed/channels#level2-batch-channel)

```text
10:52:41 [INFO] data.feeds.base_feed: Slow Coinbase receipt epoch_ms=1795670561184.195 latency_ms=1275.300 clock={'offset_ms': 9.891986846923828, 'rtt_ms': 3.2106516882777214, 'method': 'NTP:in.pool.ntp.org', 'error_bound_ms': 67.84372916445136} raw={"type":"l2update","product_id":"BTC-USD","changes":[["buy","53258.61","0.01548512"],["buy","60711.70","0.00000000"],["buy","60712.76","0.00000000"],["buy","60866.98","0.00018870"],["sell","60889.40","0.60423200"]],"time":"2026-11-26T05:22:39.918780Z"}
```

## Test output

`python -B -m pytest tests -p no:cacheprovider`

```text
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\SIDDHU\.gemini\antigravity\scratch\arbsense
plugins: anyio-4.12.1
collected 66 items

tests\test_arrival_audit.py ............                                 [ 18%]
tests\test_basis_hysteresis_labels.py ...                                [ 22%]
tests\test_engine.py ...........                                         [ 39%]
tests\test_feeds.py ...                                                  [ 43%]
tests\test_harden_refactor.py ............                               [ 62%]
tests\test_long_run_freeze.py ....                                       [ 68%]
tests\test_models.py ..........                                          [ 83%]
tests\test_normalizer.py ......                                          [ 92%]
tests\test_storage.py .....                                              [100%]

======================== 66 passed in 88.78s (0:01:28) ========================
```
