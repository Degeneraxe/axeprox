# Connection warm-up, book freshness and SOL admission · 26 November 2026

The original v2.3 ten-minute smoke database was read-only for these checks. No new collector session ran. The SOL program used separate public WebSocket and REST connections.

## Smoke event diagnosis

Every raw top-of-book update was retained in this capture, including updates whose depth was throttled. For each missing event the audit selected the latest received quote at or before arrival on the two native legs. A quote can violate more than one rule. The primary-cause table checks absence first, then local age, book validity and timestamp alignment. All four over-age cases also violate alignment. The other case violates alignment alone. The oldest Binance leg was 883.27 ms old. The five cases provide no basis for loosening the existing freshness rules.


```text
symbol  buy sell   buy_age_ms sell_age_ms separation_ms
ETH/USDT binance kraken 432.14 1.8 430.35
ETH/USDT binance kraken 581.81 0.37 581.44
BTC/USDT binance kraken 867.67 28.99 838.68
ETH/USDT binance kraken 883.27 54.19 829.08
ETH/USDT binance kraken 139.72 6.12 133.6
```

The new collector stores book diagnostics per side, including native pair, receipt time, receipt delay and exchange-time age when measured. For the eight previously priced events the audit matched their saved receipt timestamps back to the exact raw ticks. Estimated timestamps supply no exchange-age measurement. The 500 ms flag only describes the evidence. The exact Clopper–Pearson interval treats each priced event as a binomial trial. The time-cluster bootstrap is included because outcomes may be dependent.


`python scripts/audit_capture.py smoke600.db --seconds 600`

```text
raw_rows=290928 seconds=600 rows/s=484.88 DB_bytes=90513408 MB/min=9.051 gross_GB/day=13.034
permanent_upper_GB/day=0.231 projected_5day_DB_GB=4.357 replay_hours=6
missing_books primary cause (only affected routes)
symbol buy sell no_book older_than_250ms more_than_100ms_apart crossed_or_empty connection_warmup unknown
BTC/USDT binance kraken 0 1 0 0 0 0
ETH/USDT binance kraken 0 3 1 0 0 0
exchange age >500ms: would drop 0 of 8 priced events; unknown=0
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
retail: 0 of 8 priced events; exact 95% CI [0.00000, 0.36942]
retail 8 -15.13660 -15.05582 -14.50772 -4.07573 -4.07410 -10.76483 0.00000 0.00000 0.00000
vip1: 0 of 8 priced events; exact 95% CI [0.00000, 0.36942]
vip1 8 -9.95333 -9.87058 -9.32748 -2.63163 -2.62936 -6.98376 0.00000 0.00000 0.00000
vip3: 0 of 8 priced events; exact 95% CI [0.00000, 0.36942]
vip3 8 -6.59948 -6.51547 -5.97539 -1.69725 -1.69458 -4.53718 0.00000 0.00000 0.00000
zero_maker: 0 of 8 priced events; exact 95% CI [0.00000, 0.36942]
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

## Latency after warm-up

New captures mark every raw tick with the global connection gate, including reconnect warm-ups. The older smoke capture had one session with zero reported reconnects. The legacy calculation uses the last venue's first recorded quote plus 60 seconds as a conservative proxy. The input is inbound transit for retained exchange-timestamped ticks. The 85 ms viability threshold is unchanged pending the VM capture.

`python scripts/propose_latency_threshold.py --db smoke600.db --legacy-single-session`

```text
================================================================================
ArbSense Empirical Latency Threshold Analysis Tool
Database Target: C:/Users/SIDDHU/Documents/Codex/2026-11-24/referenced-chatgpt-conversation-this-is-an/work/smoke600.db
================================================================================
Warm-up filter: legacy latest first receipt + 60 s
clock_error_bound_ms 2026-11-26T05:22:38.272399+00:00 NTP:in.pool.ntp.org 67.84372916445136
clock_error_bound_ms 2026-11-26T05:27:53.407502+00:00 NTP:pool.ntp.org 55.533789563924074
clock_error_bound_ms 2026-11-26T05:32:38.970841+00:00 NTP:pool.ntp.org 67.93787516653538

[Tick Source Breakdown]
Total Ticks: 290,928
  - estimated: 16,116 (5.5%)
  - exchange: 274,812 (94.5%)

[Empirical Latency Statistics (ts_source='exchange')]
Scope              Count      Min      p50      p75      p90      p95      p99     Mean      Std
--------------------------------------------------------------------------------------------
kraken            226946    105.1    144.8    206.4    322.1    441.6    736.1    191.4    124.3
coinbase           18603    113.7    140.8    162.1    198.4    261.4    534.5    160.1     74.4
--------------------------------------------------------------------------------------------
OVERALL           245549    105.1    144.1    201.3    311.1    433.8    716.1    189.0    121.5
--------------------------------------------------------------------------------------------
These are inbound market-data delays. The VM execution threshold is set separately.
```

## SOL REST comparison

SOL was removed from the dataset's supported pairs. Kraken SOL/USDT exceeded 3 bps in 8 of 148 checks (5.41%). The admission rule requires fewer than 3%. The first session lost connectivity. Only its uninterrupted prefix was retained. A second session completed the required span. The CSV elapsed time combines the two active sampling windows and excludes the interruption. Each product requires at least 100 paired observations spanning 1,800 seconds or more. The existing acceptance criteria are mean bid/ask difference below 2 bps and fewer than 3% of observations with either side differing by more than 3 bps. A REST response is compared with the latest locally available WebSocket quote when it returns. Transport delay and intrainterval market movement can contribute to differences. [All paired quotes](2026-11-26-sol-rest-checks.csv)

`python -B -u scripts/dev/audit_sol.py --resume-log sol-rest-followup.log`

```text
RESULT {"venue": "binance", "pair": "solusdt", "checks": 148, "span_s": 1807.29, "mean_bps": 0.0, "over3": 0, "passed": true}
RESULT {"venue": "kraken", "pair": "SOL/USDT", "checks": 148, "span_s": 1807.23, "mean_bps": 0.6888996996691675, "over3": 8, "passed": false}
RESULT {"venue": "coinbase", "pair": "SOL-USD", "checks": 148, "span_s": 1807.34, "mean_bps": 0.34250914781294944, "over3": 3, "passed": true}
RESULT {"venue": "coinbase", "pair": "SOL-USDT", "checks": 148, "span_s": 1807.4, "mean_bps": 0.33272559628223, "over3": 1, "passed": true}
```
