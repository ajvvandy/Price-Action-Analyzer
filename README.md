# Brooks Price Action Analyzer

An Al-Brooks–style price-action tool for intraday charts that a basic screener cannot replicate.  
Upload a 5-minute OHLCV CSV and get bar-by-bar context and signals:

- Always-In Direction (bull, bear, neutral)
- Microchannels (len of higher-low or lower-high runs)
- H1/H2 and B1/B2 pullback markers from swing logic
- Wedges (three pushes) as exhaustion context
- Trading-Range score (overlap and failed breakout frequency)
- Breakout follow-through quality
- Measured-move target estimates
- Tradability+ score (0–100) combining context, signal quality, and range penalty

**Why this is different from a screener.**  
It scores **context and expectancy** instead of raw indicator values: microchannel persistence, pullback count quality (H1/H2 vs B1/B2), breakout follow-through rate, range re-entry, and Always-In alignment. Screeners typically cannot encode these bar-sequence patterns.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a CSV with columns: `Datetime, Open, High, Low, Close, Volume` (5-minute bars).

## Data

Export intraday data from your trading platform as CSV. Ensure times cover the regular session. The analyzer is timeframe-agnostic but thresholds are tuned for 5-minute.

## Notes

- Educational and research use only. Not financial advice.
- Heuristics approximate Al Brooks methods and can be tuned in `brooks_price_action.py`.
- For 1-minute bars, adjust microchannel and window lengths upward.

## Resume line

Built and deployed an Al-Brooks style **Price-Action Analyzer** that derives Always-In direction, microchannels, H1/H2 and B1/B2 setups, wedge and trading-range diagnostics, breakout follow-through probability, and a composite **Tradability+** score from intraday OHLCV; Streamlit app with bar-level signals and measured-move targets.
