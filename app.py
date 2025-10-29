from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Brooks Price-Action Day Report", layout="wide")
st.title("Brooks Price-Action Day Report")
st.caption("Enter a ticker. App fetches recent 5-minute data and summarizes Brooks-style context. Educational use only.")

# ----------------------- UI -----------------------
col1, col2 = st.columns([2, 1])
symbol = col1.text_input("Symbol", value="AAPL").strip().upper()
go = col2.button("Analyze")

# ------------------ Helpers / I/O ------------------
def normalize_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    out = df.reset_index().copy()
    dt_col = next((c for c in ["Datetime", "Date", "index"] if c in out.columns), None)
    if dt_col is None:
        return None

    out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce")
    try:
        out[dt_col] = out[dt_col].dt.tz_localize(None)
    except Exception:
        pass

    out = out.rename(columns={dt_col: "Datetime"})
    needed = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    if not all(c in out.columns for c in needed):
        return None

    out = out[needed].dropna().sort_values("Datetime").drop_duplicates(subset=["Datetime"])

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    return out

@st.cache_data(show_spinner=False, ttl=300)
def fetch_5m(sym: str) -> Optional[pd.DataFrame]:
    tries = [("5m", "7d"), ("5m", "30d"), ("15m", "60d")]
    for interval, period in tries:
        try:
            raw = yf.download(sym, interval=interval, period=period,
                              auto_adjust=False, progress=False, threads=False)
        except Exception:
            continue
        df = normalize_ohlcv(raw)
        if df is not None and not df.empty:
            df.attrs["interval"] = interval
            df.attrs["period"] = period
            return df
    return None

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

# ------------------ Brooks-style metrics ------------------
def overlap_score(df: pd.DataFrame, window: int = 20) -> pd.Series:
    hh = df["High"].rolling(window).max()
    ll = df["Low"].rolling(window).min()
    width = (hh - ll).replace(0, np.nan)
    mid = (hh + ll) / 2.0
    mid_time = ((df["Close"] - mid).abs() < 0.2 * width).rolling(window).mean()
    fail_bo = (((df["Close"] > hh.shift(1)) & (df["Close"].shift(1) < hh.shift(2))) |
               ((df["Close"] < ll.shift(1)) & (df["Close"].shift(1) > ll.shift(2)))).rolling(window).mean()
    sc = (0.7 * mid_time.fillna(0) + 0.3 * fail_bo.fillna(0)).clip(0, 1)
    return sc

def microchannel_lengths(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Scalar-safe microchannel counter using NumPy arrays."""
    lows = df["Low"].to_numpy()
    highs = df["High"].to_numpy()
    n = len(lows)
    bull_run = np.zeros(n, dtype=int)
    bear_run = np.zeros(n, dtype=int)
    b = s = 0
    for i in range(n):
        if i > 0 and lows[i] > lows[i - 1]:
            b += 1
        else:
            b = 0
        if i > 0 and highs[i] < highs[i - 1]:
            s += 1
        else:
            s = 0
        bull_run[i] = b
        bear_run[i] = s
    return pd.Series(bull_run, index=df.index), pd.Series(bear_run, index=df.index)

def always_in(df: pd.DataFrame) -> Tuple[str, pd.Series, pd.Series]:
    """EMA20/EMA50 + two-bar breakout bias. Scalar-safe with NumPy arrays."""
    if len(df) < 3:
        return "neutral", ema(df["Close"], 20), ema(df["Close"], 50)

    ema20 = ema(df["Close"], 20)
    ema50 = ema(df["Close"], 50)
    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    e50 = ema50.to_numpy()

    direction = "neutral"
    for i in range(2, len(close)):
        c = float(close[i])
        c1 = float(close[i - 1])
        h1 = float(high[i - 1]); h2 = float(high[i - 2])
        l1 = float(low[i - 1]);  l2 = float(low[i - 2])
        e = float(e50[i])

        bo_up = (c > h1) and (c1 > h2)
        bo_dn = (c < l1) and (c1 < l2)

        if bo_up and c > e:
            direction = "bull"
        elif bo_dn and c < e:
            direction = "bear"
        else:
            if direction == "bull" and c < e:
                direction = "neutral"
            elif direction == "bear" and c > e:
                direction = "neutral"

    return direction, ema20, ema50

def bar18_flag(day: pd.DataFrame) -> Tuple[bool, Dict]:
    n = len(day)
    if n < 20:
        return False, {}
    i0, i1 = 15, min(19, n - 1)  # bars 16..20
    seg = day.iloc[i0:i1 + 1].copy()
    rng = (seg["High"] - seg["Low"]).replace(0, np.nan)
    body = (seg["Close"] - seg["Open"])
    body_pct = (body.abs() / rng).fillna(0.0)
    mc_bull, mc_bear = microchannel_lengths(day)
    ema20_full = ema(day["Close"], 20)
    stretch = (seg["Close"] - ema20_full.iloc[i0:i1 + 1]).abs() / day["Close"].iloc[i0:i1 + 1]
    micro_ok = int(max(mc_bull.iloc[i1], mc_bear.iloc[i1])) >= 6
    body_ok = float(body_pct.mean()) >= 0.55
    stretch_ok = float(stretch.mean()) >= 0.01
    flag = bool(micro_ok or (body_ok and stretch_ok))
    details = {
        "bars_considered": f"{i0 + 1}-{i1 + 1}",
        "microchannel_len": int(max(mc_bull.iloc[i1], mc_bear.iloc[i1])),
        "avg_body_pct": round(float(body_pct.mean()), 3),
        "avg_stretch_vs_ema20": round(float(stretch.mean()), 4),
    }
    return flag, details

def measured_move(day: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    """Return (MM_up, MM_down). None means 'no estimate'."""
    if len(day) < 20:
        return None, None
    first_hour = day.iloc[:12]
    uptrend = first_hour["Close"].iloc[-1] > first_hour["Open"].iloc[0]
    if uptrend:
        leg = float(first_hour["High"].max() - day["Open"].iloc[0])
        seg = day.iloc[12:20]
        if seg.empty:
            return None, None
        pb_end_idx = seg["Low"].rolling(4).min().idxmin()
        target = float(day.loc[pb_end_idx, "Close"] + leg)
        return target, None
    else:
        leg = float(day["Open"].iloc[0] - first_hour["Low"].min())
        seg = day.iloc[12:20]
        if seg.empty:
            return None, None
        pb_end_idx = seg["High"].rolling(4).max().idxmax()
        target = float(day.loc[pb_end_idx, "Close"] - leg)
        return None, target

def opening_range_breakout(day: pd.DataFrame) -> Optional[Dict]:
    if len(day) < 8:
        return None
    orange = day.iloc[:6]  # first 30m
    hi, lo = float(orange["High"].max()), float(orange["Low"].min())
    after = day.iloc[6:]
    if after.empty:
        return {"direction": "none", "level": None, "follow_through": None}
    bo_up = after["Close"] > hi
    bo_dn = after["Close"] < lo
    if bo_up.any():
        idx = bo_up.idxmax()
        ft = bool(day.loc[idx:].head(2)["Close"].iloc[-1] > day.loc[idx, "Close"])
        return {"direction": "up", "level": hi, "follow_through": ft}
    if bo_dn.any():
        idx = bo_dn.idxmax()
        ft = bool(day.loc[idx:].head(2)["Close"].iloc[-1] < day.loc[idx, "Close"])
        return {"direction": "down", "level": lo, "follow_through": ft}
    return {"direction": "none", "level": None, "follow_through": None}

def range_vs_adr(day: pd.DataFrame, hist: pd.DataFrame) -> Tuple[float, Optional[float]]:
    try:
        adr = (hist["High"] - hist["Low"]).resample("1D").max().dropna().tail(14).mean()
        adr = float(adr) if pd.notna(adr) else None
    except Exception:
        adr = None
    today_r = float(day["High"].max() - day["Low"].min())
    return today_r, adr

# ----------------------- Analysis -----------------------
if go and symbol:
    with st.spinner(f"Fetching {symbol} 5-minute data…"):
        df_all = fetch_5m(symbol)

    if df_all is None or df_all.empty:
        st.error("Could not fetch data. Try another symbol or later.")
    else:
        df_all["Date"] = df_all["Datetime"].dt.date
        last_date = df_all["Date"].max()
        day = df_all[df_all["Date"] == last_date].copy()
        if day.empty and len(df_all) > 0:
            day = df_all.tail(78).copy()
        day = day.drop(columns=["Date"])

        required_cols = {"Open", "High", "Low", "Close", "Volume", "Datetime"}
        if not required_cols.issubset(set(day.columns)):
            st.error("Missing required columns after fetch. Upload a CSV instead.")
            st.stop()
        if len(day) < 10:
            st.error("Not enough intraday bars to analyze this session.")
            st.stop()

        o = float(day["Open"].iloc[0])
        c = float(day["Close"].iloc[-1])
        hi = float(day["High"].max())
        lo = float(day["Low"].min())
        pct_from_open = 100 * (c - o) / o
        pct_to_high = 100 * (hi - c) / c
        pct_to_low = 100 * (c - lo) / c

        always, ema20, ema50 = always_in(day)
        mc_bull, mc_bear = microchannel_lengths(day)
        overlap = float(overlap_score(day, window=24).iloc[-1])
        bar18, b18d = bar18_flag(day)
        mm_up, mm_dn = measured_move(day)
        or_bo = opening_range_breakout(day)
        today_range, adr14 = range_vs_adr(day, df_all.set_index("Datetime"))

        flags = []
        if always == "bull": flags.append("Always-In Bull")
        elif always == "bear": flags.append("Always-In Bear")
        if overlap >= 0.55: flags.append("Trading-Range Day")
        if max(int(mc_bull.iloc[-1]), int(mc_bear.iloc[-1])) >= 6: flags.append("Microchannel ≥6")
        if or_bo and or_bo.get("direction") == "up": flags.append("Opening BO Up")
        if or_bo and or_bo.get("direction") == "down": flags.append("Opening BO Down")
        if bar18: flags.append("Bar-18 Exhaustion Risk")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("% from Open", f"{pct_from_open:.2f}%")
        m2.metric("% to Session High", f"{pct_to_high:.2f}%")
        m3.metric("% to Session Low", f"{pct_to_low:.2f}%")
        m4.metric("Day Range vs ADR14", f"{today_range:.2f} / {adr14:.2f}" if adr14 is not None else f"{today_range:.2f} / —")

        n1, n2, n3 = st.columns(3)
        n1.metric("Always-In", always)
        n2.metric("Trading-Range Score (0–1)", f"{overlap:.2f}")
        n3.metric("Microchannel len (bull/bear)", f"{int(mc_bull.iloc[-1])}/{int(mc_bear.iloc[-1])}")

        st.subheader("Session Flags")
        st.write(", ".join(flags) if flags else "No special conditions flagged.")

        st.subheader("Bar-18 Heuristic (bars 16–20)")
        st.json({**b18d, "flag": bool(bar18)})

        st.subheader("Opening-Range Breakout")
        st.json(or_bo if or_bo else {"direction": "none"})

        st.subheader("Measured-Move Targets (approx)")
        st.write({"MM_up": mm_up, "MM_down": mm_dn})

        tbl = day.copy()
        tbl["ema20"] = ema20.values
        tbl["ema50"] = ema50.values
        tbl["bull_mc_len"] = mc_bull.values
        tbl["bear_mc_len"] = mc_bear.values
        st.subheader("Bars (latest 120)")
        st.dataframe(tbl.tail(120), use_container_width=True)

        st.subheader("Chart (Close with EMA20/50)")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(day["Datetime"], day["Close"], label="Close")
        ax.plot(day["Datetime"], ema20, label="EMA20")
        ax.plot(day["Datetime"], ema50, label="EMA50")
        ax.set_xlabel("Time"); ax.set_ylabel("Price"); ax.legend(loc="best")
        st.pyplot(fig)

        st.download_button(
            "Download Day Bars CSV",
            day.to_csv(index=False).encode(),
            file_name=f"{symbol}_{last_date}_5m.csv",
            mime="text/csv")
