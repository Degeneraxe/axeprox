# BTC/ETH size and delay replay

The archived ten-minute v2.3 capture contained 13 native Binance/Kraken BTC/ETH events, in both directions combined. The other 31 events required currency conversion. Replay covers all 13 original native candidates, including startup, across six delays (25, 50, 85, 100, 250 and 500 ms), five decision notionals (50, 100, 250, 500 and 1,000 USDT) and the four hypothetical fee schedules frozen in the capture. These are alternative paper outcomes for the same observations.

At each event opening the requested buy notional determines the base quantity, rounded down to a common valid lot. Both legs must meet quantity and notional limits. Market orders take recorded prices. The pair limits are a later public snapshot, with its retrieval time and sources saved in [the limits file](smoke-market-limits.json). No account-specific fee schedule was available for this experiment.

Each leg uses its latest book received by simulated arrival. The original 250 ms age and 100 ms alignment rules apply. The best bid and ask give an optimistic upper bound on after-fee P&L even when depth is missing. For actual simulated P&L the replay requires sufficient recorded depth on both legs from those exact receipts. It can reuse the original arrival evidence when the receipt timestamps match. Depth snapshots were throttled to 100 ms, which limits exact pricing at other delays. The P&L sum is blank where no events could be priced.

At no tested delay did a fresh best-price bound become positive under any configured fee scenario. At 85 ms all eight fresh events could be depth-priced at every target size. VIP3 total P&L was -0.7973, -1.5983, -4.0087, -8.0708 and -16.2836 USDT as size increased. Increasing size increased absolute losses in this capture. The ten-minute result leaves longer-run profitability open.

Run `python scripts/replay_size_delay.py path/to/smoke600.db docs/smoke-size-delay.csv` on the original unpruned smoke capture. [All 120 scenario results](smoke-size-delay.csv)
