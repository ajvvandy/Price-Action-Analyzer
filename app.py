import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from brooks_price_action import load_csv, analyze_session

st.set_page_config(page_title="Brooks Price Action Analyzer", layout="wide")
st.title("Brooks Price Action Analyzer")

st.write("Enter a stock symbol (e.g., AAPL, NVDA, TSLA) **or** upload a 5-minute CSV. Educational use only.")

# --- UI inputs
default_list = ["AAPL","MSFT","NVDA","META","AMZN","TSLA","AMD","SPY","QQQ"]
c1, c2 = st.columns([2,1])
ticker = c1.text_input("Stock Symbol", value=default_list[0]).strip().upper()
uploaded = c2.file_uploader("Or upload CSV", type=["csv"])

# --- helper: normalize yfinance frame to required columns
def normalize_ohlcv(df):
    """
    yfinance intraday returns index as DatetimeIndex with tz and columns:
    Open, High, Low, Close, Adj Close, Volume.
    We need columns: Datetime, Open, High, Low, Close, Volume
    """
    if df is None or df.empty:
        return None
    # Reset index to get a Datetime column, drop adj close, remove tz for safety
    out = df.reset_index().copy()
    # Figure out the datetime column name: often 'Datetime' but can be 'index'
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
        # already tz-naive or non-datetime; ignore
        pass

    # Standardize headers
    rename_map = {
        dt_col: "Datetime",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "Adj Close",
        "Volume": "Volume",
    }
    out = out.rename(columns=rename_map)
    # Keep only needed columns
    needed = ["Datetime","Open","High","Low","Close","Volume"]
    if not all(c in out.columns for c in needed):
        return None
    out = out[needed].dropna()
    # Sometimes yfinance returns duplicate or unsorted rows
    out = out.sort_values("Datetime").drop_duplicates(subset=["Datetime"])
    return out

# --- helper: fetch with fallback intervals
@st.cache_data(show_spinner=False, ttl=300)
def fetch_intraday(symbol: str):
    """
    Try 5m with 7d period.
    If empty, fall back to 15m with 30d period.
    Return normalized OHLCV or None.
    """
    # 5m supports up to 30d, but 7d is usually reliable
    tries = [
        dict(interval="5m", period="7d"),
        dict(interval="15m", period="30d"),
    ]
    for t in tries:
        try:
            df = yf.download(symbol, interval=t["interval"], period=t["period"], auto_adjust=False, progress=False, threads=False)
            norm = normalize_ohlcv(df)
            if norm is not None and not norm.empty:
                norm.attrs["interval"] = t["interval"]
                norm.attrs["period"] = t["period"]
                return norm
        except Exception:
            continue
    return None

# --- decide data source
df = None
source = None

if uploaded is not None:
    # CSV path
    try:
        df = load_csv(uploaded)  # already normalizes headers
        source = "csv"
    except Exception as e:
        st.error(f"CSV error: {e}")

elif ticker:
    with st.spinner(f"Fetching data for {ticker}…"):
        fetched = fetch_intraday(ticker)
    if fetched is None or fetched.empty:
        st.error("Could not fetch intraday data. Check the symbol, or try uploading a CSV.")
    else:
        df = fetched
        interval = df.attrs.get("interval", "unknown")
        period = df.attrs.get("period", "unknown")
        st.caption(f"Fetched {ticker} ({interval}, {period}) — {len(df)} bars")

# --- run analysis if we have data
if df is not None and not df.empty:
    try:
        sig, ctx, extras = analyze_session(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Always-In", ctx.always_in)
        col2.metric("Trading Range Score (0–1; lower better)", f"{ctx.trading_range_score:.2f}")
        col3.metric("Tradability+ (0–100)", f"{extras['tradability_plus']:.1f}")

        st.subheader("Signals (last 150 bars)")
        st.dataframe(sig.tail(150), use_container_width=True)

        st.subheader("Chart preview")
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
    st.info("Enter a valid symbol (e.g., AAPL) or upload a CSV with columns: Datetime, Open, High, Low, Close, Volume.")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from brooks_price_action import load_csv, analyze_session

st.set_page_config(page_title="Brooks Price Action Analyzer", layout="wide")
st.title("Brooks Price Action Analyzer")

st.write("Upload 5-minute OHLCV CSV to compute Al-Brooks style context and signals. Educational use only.")

import yfinance as yf

st.write("Enter a stock symbol (e.g. AAPL, NVDA, TSLA) or upload your own CSV.")

ticker = st.text_input("Stock Symbol", value="AAPL")
uploaded = st.file_uploader("Or upload CSV (optional)", type=["csv"])

if ticker and not uploaded:
    st.info(f"Fetching 5-minute data for {ticker}...")
    df = yf.download(ticker, period="5d", interval="5m")
    df = df.reset_index()
    df = df.rename(columns={"Datetime": "Datetime", "Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
elif uploaded:
    df = load_csv(uploaded)
else:
    df = None

if uploaded is not None:
    try:
        df = load_csv(uploaded)
        sig, ctx, extras = analyze_session(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Always-In", ctx.always_in)
        col2.metric("Trading Range Score (0-1 lower better)", f"{ctx.trading_range_score:.2f}")
        col3.metric("Tradability+ (0-100)", f"{extras['tradability_plus']:.1f}")

        st.subheader("Signals table (last 100 bars)")
        st.dataframe(sig.tail(100))

        st.subheader("Chart preview")
        fig, ax = plt.subplots(figsize=(12,4))
        dft = df.copy()
        dft['Datetime'] = pd.to_datetime(dft['Datetime'])
        ax.plot(dft['Datetime'], dft['Close'], label='Close')
        ax.plot(sig.index, sig['ema20'], label='EMA20')
        ax.plot(sig.index, sig['ema50'], label='EMA50')
        ax.legend(loc="best")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        st.pyplot(fig)

        st.subheader("Measured-Move Targets")
        st.json(extras)

        st.download_button("Download Signals CSV", sig.to_csv().encode(), "signals.csv", "text/csv")

    except Exception as e:
        st.error(str(e))
else:
    st.caption("Tip: export 5-minute data from your platform as CSV and upload here.")
