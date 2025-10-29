import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

from brooks_price_action import analyze_session, load_csv

st.set_page_config(page_title="Brooks Price Action Analyzer", layout="wide")
st.title("Brooks Price Action Analyzer")

# --------- Helpers

def parse_symbol_from_tv_url(url: str) -> str | None:
    """
    Parse a TradingView chart URL and extract the 'symbol' param.
    Examples:
      https://www.tradingview.com/chart/AbCdEf/?symbol=NASDAQ%3AAAPL
      https://www.tradingview.com/chart/?symbol=NYSE%3ABRK.B
    """
    if not url:
        return None
    try:
        from urllib.parse import urlparse, parse_qs, unquote
        qs = parse_qs(urlparse(url).query)
        if "symbol" in qs and qs["symbol"]:
            raw = qs["symbol"][0]
            sym = unquote(raw)
            # common TradingView style "NASDAQ:AAPL" → we just need "AAPL"
            if ":" in sym:
                sym = sym.split(":")[-1]
            return sym.upper().strip()
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False, ttl=300)
def fetch_intraday(symbol: str):
    """
    Try 5m/7d; if empty, fall back to 15m/30d.
    Normalize columns to: Datetime, Open, High, Low, Close, Volume.
    """
    def normalize(df):
        if df is None or df.empty:
            return None
        out = df.reset_index().copy()
        # datetime column name can vary
        dt_col = None
        for cand in ["Datetime", "Date", "index"]:
            if cand in out.columns:
                dt_col = cand
                break
        if dt_col is None:
            return None
        out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce")
        try:
            out[dt_col] = out[dt_col].dt.tz_localize(None)
        except Exception:
            pass
        out = out.rename(columns={
            dt_col: "Datetime",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        })
        need = ["Datetime","Open","High","Low","Close","Volume"]
        if not all(c in out.columns for c in need):
            return None
        out = out[need].dropna().sort_values("Datetime").drop_duplicates(subset=["Datetime"])
        return out

    tries = [
        dict(interval="5m", period="7d"),
        dict(interval="15m", period="30d"),
    ]
    for t in tries:
        try:
            df = yf.download(symbol, interval=t["interval"], period=t["period"], auto_adjust=False, progress=False, threads=False)
            norm = normalize(df)
            if norm is not None and not norm.empty:
                norm.attrs["interval"] = t["interval"]
                norm.attrs["period"] = t["period"]
                return norm
        except Exception:
            continue
    return None

def tv_widget_html(symbol: str, height: int = 520):
    """
    TradingView widget embed (client-side only).
    This does NOT give us data; it's just a live chart view in the app.
    """
    sym = symbol.upper()
    return f"""
    <div class="tradingview-widget-container">
      <div id="tvchart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "width": "100%",
          "height": {height},
          "symbol": "{sym}",
          "interval": "5",
          "timezone": "Etc/UTC",
          "theme": "light",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#f1f3f6",
          "enable_publishing": false,
          "allow_symbol_change": true,
          "container_id": "tvchart"
        }});
      </script>
    </div>
    """

# --------- Read symbol from query param (so bookmarklet/link works)

qp = st.query_params
symbol_from_qs = None
if "symbol" in qp and qp["symbol"]:
    # qp["symbol"] can be str or list
    symbol_from_qs = qp["symbol"]
    if isinstance(symbol_from_qs, list):
        symbol_from_qs = symbol_from_qs[0]
    symbol_from_qs = symbol_from_qs.upper()

# --------- UI: three ways to select data source

st.write("Enter a stock symbol, paste a TradingView chart URL (we’ll parse the symbol), **or** upload a CSV (5-minute OHLCV).")

c1, c2 = st.columns([2, 2])

default_symbol = symbol_from_qs or "AAPL"
symbol = c1.text_input("Stock Symbol", value=default_symbol).strip().upper()

tv_url = c2.text_input("TradingView Chart URL (optional, we parse the symbol)", value="", placeholder="https://www.tradingview.com/chart/…?symbol=NASDAQ%3AAAPL")

uploaded = st.file_uploader("Or upload CSV (columns: Datetime, Open, High, Low, Close, Volume)", type=["csv"])

# If user pasted a TV URL, override symbol.
if tv_url:
    parsed = parse_symbol_from_tv_url(tv_url)
    if parsed:
        symbol = parsed
        st.success(f"Parsed symbol from TradingView URL: {symbol}")
    else:
        st.warning("Could not parse symbol from TradingView URL. Using the symbol field instead.")

# --------- Data selection logic

df = None
source = None

if uploaded is not None:
    try:
        df = load_csv(uploaded)
        source = "csv"
    except Exception as e:
        st.error(f"CSV error: {e}")

elif symbol:
    with st.spinner(f"Fetching intraday data for {symbol}…"):
        fetched = fetch_intraday(symbol)
    if fetched is None or fetched.empty:
        st.error("Could not fetch intraday data. Check the symbol or upload a CSV.")
    else:
        df = fetched
        source = f"yfinance:{symbol}"
        st.caption(f"Fetched {symbol} ({df.attrs.get('interval','?')} interval, {df.attrs.get('period','?')} period) — {len(df)} bars")

# --------- Show embedded TradingView chart (visual only)

if symbol:
    st.subheader("TradingView chart (embedded)")
    st.components.v1.html(tv_widget_html(symbol), height=540, scrolling=False)

# --------- Analysis

if df is not None and not df.empty:
    try:
        sig, ctx, extras = analyze_session(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Always-In", ctx.always_in)
        col2.metric("Trading Range Score (0–1; lower better)", f"{ctx.trading_range_score:.2f}")
        col3.metric("Tradability+ (0–100)", f"{extras['tradability_plus']:.1f}")

        st.subheader("Signals (last 150 bars)")
        st.dataframe(sig.tail(150), use_container_width=True)

        st.subheader("Chart preview (EMAs)")
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df["Datetime"], df["Close"], label="Close")
        ax.plot(sig.index, sig["ema20"], label="EMA20")
        ax.plot(sig.index, sig["ema50"], label="EMA50")
        ax.legend(loc="best")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        st.pyplot(fig)

        st.subheader("Measured-Move Targets")
        st.json(extras)

        st.download_button("Download Signals CSV", sig.to_csv().encode(), "signals.csv", "text/csv")

    except Exception as e:
        st.error(f"Analysis error: {e}")
else:
    st.info("Provide a symbol, paste a TradingView chart URL, or upload a CSV.")
