Brooks Price Action Analyzer

A quantitative interpretation of Al Brooks–style price action, built to extract context that standard screeners cannot.
This tool ingests 5-minute OHLCV data and computes structural features of intraday behavior—bar-sequence logic, contextual bias, and measured-move projections—rather than indicator overlays.

Overview

Unlike conventional scanners that flag single-bar conditions, the Brooks Analyzer evaluates bar sequences and contextual transitions.
It interprets each session through the same heuristics used by discretionary price-action traders:

Always-In Direction — determines session bias (bull/bear/neutral) from bar structure and EMA alignment.

Microchannel Detection — counts consecutive higher-lows or lower-highs to assess trend persistence.

H1/H2 / B1/B2 Identification — classifies pullback legs and second-entries based on swing logic.

Wedge Recognition — detects three-push exhaustion patterns.

Trading-Range Score — quantifies overlap and failed breakout frequency (0 = trend-day, 1 = range-day).

Breakout Follow-Through — evaluates bar-to-bar momentum and pullback depth after range exits.

Measured-Move Targets — projects first-leg magnitude for exhaustion analysis.

Tradability + Score (0 – 100) — composite metric integrating trend strength, range structure, and volatility efficiency.

Why It’s Different

Screeners measure conditions (e.g., RSI, EMA crossovers).
This analyzer measures context and expectancy — how bars evolve into structure.
It captures information screeners can’t express:
trend maturity (bar 18 exhaustion), quality of pullbacks, frequency of reversals inside ranges, and whether breakouts expand or fail.
