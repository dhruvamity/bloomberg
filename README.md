# bloomberg

News-impact and Smart Money Concepts (SMC) study on gold: headlines from the Walter Bloomberg Telegram channel mapped minute by minute against Binance **XAUUSDT** perpetual on 1m, 5m, 15m, 1h, 4h and 1D.

## What's here

| Path | What it is |
|---|---|
| `gold_news_shock_map.html` | Interactive report: impact by news type and timeframe, spike-candle attribution, replay of the 40 worst events, volatility clock, playbook |
| `data/xauusdt_1m_binance.csv` | XAUUSDT 1-minute klines from listing (11 Dec 2025) onward |
| `analysis/category_impact_summary.csv` | Per-news-type range multiples, both-stops-hit rates, shock rates |
| `analysis/smc_*.csv` | SMC backtest trades (2026) and the week of 2 Nov 2026, with every setup and why it was taken or rejected |
| `scripts/` | Everything needed to rebuild the above |

Headline text is not committed. Export the channel from Telegram (HTML) and regenerate it locally.

## Rebuild

```bash
python3 scripts/parse.py <telegram_export_dir> data/walter_bloomberg_headlines_utc.csv
python3 scripts/fetch.py data/xauusdt_1m_binance.csv
python3 scripts/analyze.py data/xauusdt_1m_binance.csv data/walter_bloomberg_headlines_utc.csv analysis
python3 scripts/build_report.py analysis/report_data.json gold_news_shock_map.html
```

## SMC tools

- `scripts/smc_backtest.py`: rule-based model. 4h bias, liquidity sweep (previous-day high/low, Asia, London, 1h swings) in the London/NY AM killzones, MSS with displacement, FVG limit entry, stop beyond the sweep, target at opposing liquidity. No lookahead; fills and exits are resolved on 1m bars.
- `scripts/smc_live.py`: live scan of HTF structure, liquidity map, HTF FVGs and LTF setups forming now.

```bash
python3 scripts/smc_live.py data/xauusdt_1m_binance.csv
```

Research only, not financial advice.
