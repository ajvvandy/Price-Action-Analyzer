import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Brooks Price-Action Day Report", layout="wide")
st.title("Brooks Price-Action Day Report")

st.caption("Enter a ticker. App fetches recent 5-minute data and summarizes Brooks-style context. Educational use only.")

# ----------------------- UI -----------------------
col1, col2 = st.columns([2,1])
symbol = col1.text_input("Symbol", value="AAPL").strip().upper()
go = col2.button("Analyze")

# ------------------ Helpers / Indicators ------------------
def normalize_ohlcv(df):
    if df is None or df.empty: return None
    out = df.reset_index().copy()
    dt_col = next((c for c in ["Datetime","Date","index"] if c in out.columns), None)
    if dt_col is None: return None
    out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce")
    try: out[dt_col] = out[dt_col].dt.tz_localize(None)
    except Exception: pass
    out = out.rename(columns={dt_col:"Datetime"})
    need = ["Datetime","Open","High","Low","Close","Volume"]
    for c in need:
        if c not in out.columns: return None
    out = out[need].dropna().sort_values("Datetime")
    return out

@st.cache_data(show_spinner=False, ttl=300)
def fetch_5m(sym):
    tries = [("5m","7d"), ("5m","30d"), ("15m","60d")]
    for interval, period in tries:
        raw = yf.download(sym, interval=interval, period=period, auto_adjust=False, progress=False, threads=False)
        df = normalize_ohlcv(raw)
        if df is not None and not df.empty:
            df.attrs["interval"] = interval; df.attrs["period"] = period
            return df
    return None

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def true_range(df):
    pc = df["Close"].shift(1)
    tr = pd.concat([(df["High"]-df["Low"]).abs(),
                    (df["High"]-pc).abs(),
                    (df["Low"]-pc).abs()], axis=1).max(axis=1)
    return tr

def overlap_score(df, window=20):
    """0..1 high means range-like: time near mid + failed BOs."""
    hh = df["High"].rolling(window).max()
    ll = df["Low"].rolling(window).min()
    width = (hh-ll).replace(0,np.nan)
    mid = (hh+ll)/2.0
    mid_time = ((df["Close"]-mid).abs() < 0.2*width).rolling(window).mean()
    fail_bo = (((df["Close"]>hh.shift(1)) & (df["Close"].shift(1)<hh.shift(2))) |
               ((df["Close"]<ll.shift(1)) & (df["Close"].shift(1)>ll.shift(2)))).rolling(window).mean()
    sc = (0.7*mid_time.fillna(0) + 0.3*fail_bo.fillna(0)).clip(0,1)
    return sc

def microchannel_lengths(df):
    """Bull: lows above prior lows sequence; Bear: highs below prior highs."""
    bull_run, bear_run = [], []
    b, s = 0, 0
    for i in range(len(df)):
        if i>0 and df["Low"].iloc[i] > df["Low"].iloc[i-1]: b+=1
        else: b=0
        if i>0 and df["High"].iloc[i] < df["High"].iloc[i-1]: s+=1
        else: s=0
        bull_run.append(b); bear_run.append(s)
    return pd.Series(bull_run, index=df.index), pd.Series(bear_run, index=df.index)

def always_in(df):
    """Heuristic: EMA20 vs EMA50 + breakout follow-through bias."""
    ema20, ema50 = ema(df["Close"],20), ema(df["Close"],50)
    dirn = "neutral"
    for i in range(2, len(df)):
        c = df["Close"].iloc[i]
        bo_up = c > df["High"].iloc[i-1] and df["Close"].iloc[i-1] > df["High"].iloc[i-2]
        bo_dn = c < df["Low"].iloc[i-1]  and df["Close"].iloc[i-1]  < df["Low"].iloc[i-2]
        if bo_up and c > ema50.iloc[i]: dirn = "bull"
        elif bo_dn and c < ema50.iloc[i]: dirn = "bear"
        else:
            # trend maintenance
            if dirn=="bull" and c < ema50.iloc[i]: dirn = "neutral"
            if dirn=="bear" and c > ema50.iloc[i]: dirn = "neutral"
    return dirn, ema20, ema50

def bar18_flag(day):
    """
    Brooks heuristic: after ~18 bars of persistent trend, probability of exhaustion / two-leg PB rises.
    We flag if bars 16-20 show: microchannel>=6, large average body, closes near extremes, OR distance from EMA20 stretched.
    """
    n = len(day)
    if n < 20: return False, {}
    i0, i1 = 15, min(19, n-1)  # bars 16..20
    seg = day.iloc[i0:i1+1].copy()
    rng = (seg["High"]-seg["Low"]).replace(0,np.nan)
    body = (seg["Close"]-seg["Open"])
    body_pct = (body.abs()/rng).fillna(0.0)
    mc_bull, mc_bear = microchannel_lengths(day)
    stretch = (seg["Close"] - ema(day["Close"],20).iloc[i0:i1+1]).abs() / day["Close"].iloc[i0:i1+1]
    micro_ok = (max(mc_bull.iloc[i1], mc_bear.iloc[i1]) >= 6)
    body_ok  = (body_pct.mean() >= 0.55)
    stretch_ok = (stretch.mean() >= 0.01)  # ~1% from EMA20 on 5m
    flag = micro_ok or (body_ok and stretch_ok)
    details = {
        "bars_considered": f"{i0+1}-{i1+1}",
        "microchannel_len": int(max(mc_bull.iloc[i1], mc_bear.iloc[i1])),
        "avg_body_pct": round(float(body_pct.mean()),3),
        "avg_stretch_vs_ema20": round(float(stretch.mean()),4)
    }
    return bool(flag), details

def measured_move(day):
    """
    First leg from open to first significant pullback end; project from PB end.
    Simple: detect first swing extreme inside first hour and first two-leg PB end.
    """
    if len(day) < 20: return np.nan, np.nan
    first_hour = day.iloc[:12]  # 5m*12=60m
    uptrend = first_hour["Close"].iloc[-1] > first_hour["Open"].iloc[0]
    if uptrend:
        leg = first_hour["High"].max() - day["Open"].iloc[0]
        pb_end = day["Low"].rolling(4).min().iloc[12:20].idxmin()
        target = day.loc[pb_end, "Close"] + leg
        return float(target), np.nan
    else:
        leg = day["Open"].iloc[0] - first_hour["Low"].min()
        pb_end = day["High"].rolling(4).max().iloc[12:20].idxmax()
        target = day.loc[pb_end, "Close"] - leg
        return np.nan, float(target)

def opening_range_breakout(day):
    """OR = first 30 minutes; report breakout direction and 1-bar follow-through rate over last N sessions if available."""
    if len(day) < 8: return None
    orange = day.iloc[:6]  # 30m = 6 bars of 5m
    hi, lo = orange["High"].max(), orange["Low"].min()
    after = day.iloc[6:]
    bo_up = (after["Close"] > hi)
    bo_dn = (after["Close"] < lo)
    if bo_up.any():
        idx = bo_up.idxmax()
        ft = (day.loc[idx:].head(2)["Close"].iloc[-1] > day.loc[idx,"Close"])
        return {"direction":"up","level":float(hi),"follow_through":bool(ft)}
    if bo_dn.any():
        idx = bo_dn.idxmax()
        ft = (day.loc[idx:].head(2)["Close"].iloc[-1] < day.loc[idx,"Close"])
        return {"direction":"down","level":float(lo),"follow_through":bool(ft)}
    return {"direction":"none","level":None,"follow_through":None}

def range_vs_adr(day, hist):
    """Compare today's range to 14-day ADR."""
    adr = (hist["High"]-hist["Low"]).resample("1D").max().dropna().tail(14).mean()
    today_r = day["High"].max() - day["Low"].min()
    return float(today_r), float(adr) if adr==adr else np.nan

# ----------------------- Analysis -----------------------
if go and symbol:
    with st.spinner(f"Fetching {symbol} 5-minute data…"):
        df_all = fetch_5m(symbol)
    if df_all is None or df_all.empty:
        st.error("Could not fetch data. Try another symbol or later.")
    else:
        # Pick the most recent regular session (last date in data)
        df_all["Date"] = df_all["Datetime"].dt.date
        last_date = df_all["Date"].max()
        day = df_all[df_all["Date"]==last_date].copy()
        if day.empty and len(df_all)>0:
            day = df_all.tail(78).copy()  # fallback ~ full US session bars
        day = day.drop(columns=["Date"])

        # Session stats
        o = day["Open"].iloc[0]
        c = day["Close"].iloc[-1]
        hi = day["High"].max(); lo = day["Low"].min()
        pct_from_open = 100*(c - o)/o
        pct_to_high   = 100*(hi - c)/c
        pct_to_low    = 100*(c - lo)/c
        rng = hi - lo

        # Diagnostics
        always, ema20, ema50 = always_in(day)
        mc_bull, mc_bear = microchannel_lengths(day)
        overlap = overlap_score(day, window=24).iloc[-1]  # ~2 hours window
        bar18, b18d = bar18_flag(day)
        mm_up, mm_dn = measured_move(day)
        or_bo = opening_range_breakout(day)
        today_range, adr14 = range_vs_adr(day, df_all.set_index("Datetime"))

        # Quick labels
        flags = []
        if always=="bull": flags.append("Always-In Bull")
        elif always=="bear": flags.append("Always-In Bear")
        if overlap >= 0.55: flags.append("Trading-Range Day")
        if max(mc_bull.iloc[-1], mc_bear.iloc[-1]) >= 6: flags.append("Microchannel ≥6")
        if or_bo and or_bo.get("direction")=="up": flags.append("Opening BO Up")
        if or_bo and or_bo.get("direction")=="down": flags.append("Opening BO Down")
        if bar18: flags.append("Bar-18 Exhaustion Risk")

        # ----------------------- Output -----------------------
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("% from Open", f"{pct_from_open:.2f}%")
        m2.metric("% to Session High", f"{pct_to_high:.2f}%")
        m3.metric("% to Session Low", f"{pct_to_low:.2f}%")
        m4.metric("Day Range vs ADR14", f"{today_range:.2f} / {adr14:.2f}")

        n1, n2, n3 = st.columns(3)
        n1.metric("Always-In", always)
        n2.metric("Trading-Range Score (0-1)", f"{overlap:.2f}")
        n3.metric("Microchannel len (bull/bear)", f"{int(mc_bull.iloc[-1])}/{int(mc_bear.iloc[-1])}")

        st.subheader("Session Flags")
        st.write(", ".join(flags) if flags else "No special conditions flagged.")

        st.subheader("Bar-18 Heuristic (bars 16–20)")
        st.json(b18d | {"flag": bool(bar18)})

        st.subheader("Opening-Range Breakout")
        st.json(or_bo)

        st.subheader("Measured-Move Targets (approx)")
        st.write({"MM_up": mm_up, "MM_down": mm_dn})

        # Table of bar-by-bar basics
        tbl = day.copy()
        tbl["ema20"] = ema20.values
        tbl["ema50"] = ema50.values
        tbl["bull_mc_len"] = mc_bull.values
        tbl["bear_mc_len"] = mc_bear.values
        st.subheader("Bars (latest 120)")
        st.dataframe(tbl.tail(120), use_container_width=True)

        # Plot
        st.subheader("Chart (Close with EMA20/50)")
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(day["Datetime"], day["Close"], label="Close")
        ax.plot(day["Datetime"], ema20, label="EMA20")
        ax.plot(day["Datetime"], ema50, label="EMA50")
        ax.set_xlabel("Time"); ax.set_ylabel("Price"); ax.legend(loc="best")
        st.pyplot(fig)

        st.download_button("Download Day Bars CSV", day.to_csv(index=False).encode(), file_name=f"{symbol}_{last_date}_5m.csv", mime="text/csv")

