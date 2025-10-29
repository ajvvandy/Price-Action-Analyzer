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

    # 1) If yfinance gave MultiIndex columns like ('Open','AAPL'), flatten to first level.
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df.columns = [str(c[0]) if isinstance(c, tuple) and c else str(c) for c in df.columns]

    out = df.reset_index().copy()

    # 2) Find the datetime column robustly
    # common names after reset_index(): 'Datetime', 'Date', or the index name itself
    dt_candidates = ["Datetime", "Date", "date", "Index", "index"]
    dt_col = None
    for c in out.columns:
        if str(c) in dt_candidates:
            dt_col = c
            break
    # If nothing matched, but the first column looks like a datetime, use it
    if dt_col is None:
        first_col = out.columns[0]
        if np.issubdtype(pd.Series(out[first_col]).dtype, np.datetime64) or "date" in str(first_col).lower():
            dt_col = first_col

    if dt_col is None:
        return None  # cannot identify a datetime column

    # 3) Coerce to timezone-naive datetime
    out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce")
    try:
        out[dt_col] = out[dt_col].dt.tz_localize(None)
    except Exception:
        pass

    # 4) Rename datetime column to exactly 'Datetime'
    if dt_col != "Datetime":
        out = out.rename(columns={dt_col: "Datetime"})

    # 5) Ensure required price/volume columns exist (after flattening)
    # Some feeds use lowercase; normalize names first
    rename_map = {c: c.title() for c in out.columns}
    out = out.rename(columns=rename_map)

    required = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in out.columns]
    # Try alternative common names if missing
    alt_map = {}
    for c in missing:
        lc = c.lower()
        for col in out.columns:
            if col.lower() == lc:
                alt_map[col] = c
                break
    if alt_map:
        out = out.rename(columns=alt_map)

    if not all(c in out.columns for c in required):
        return None

    out = out[required]

    # 6) Enforce numeric types on OHLCV
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["Datetime", "Open", "High", "Low", "Close"]).sort_values("Datetime")

    # 7) Only de-dup on Datetime if the column truly exists
    if "Datetime" in out.columns:
        out = out.drop_duplicates(subset=["Datetime"])

    return out.reset_index(drop=True)

@st.cache_data(show_spinner=False, ttl=300)
def fetch_5m(sym: str) -> Optional[pd.DataFrame]:
    tries = [("5m", "7d"), ("5m", "30d"), ("15m", "60d")]
    last_error = None
    for interval, period in tries:
        try:
            raw = yf.download(
                sym, interval=interval, period=period,
                auto_adjust=False, progress=False, threads=False
            )
            df = normalize_ohlcv(raw)
            if df is not None and not df.empty:
                df.attrs["interval"] = interval
                df.attrs["period"] = period
                return df
        except Exception as e:
            last_error = e
            continue
    # Optional: surface a hint to the UI without crashing cache
    if last_error:
        st.warning(f"Download/normalize failed for {sym} ({last_error}).")
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


# ---------- Narrative + setups helpers ----------
def bar_number_series(df: pd.DataFrame) -> pd.Series:
    """1-based bar numbers for the session."""
    return pd.Series(np.arange(1, len(df)+1), index=df.index, name="bar")

def opening_range(day: pd.DataFrame, bars_min: int = 5, bars_max: int = 18) -> dict:
    """Compute opening-range [first N bars] hi/lo and whether current price is inside/outside."""
    n = min(len(day), bars_max)
    n = max(n, bars_min)
    or_df = day.iloc[:n]
    hi = float(or_df["High"].max())
    lo = float(or_df["Low"].min())
    mid = (hi + lo) / 2.0
    last = float(day["Close"].iloc[-1])
    status = "inside"
    if last > hi: status = "above"
    elif last < lo: status = "below"
    return {"bars": n, "high": hi, "low": lo, "mid": mid, "status": status}

def hi_lo_by_bar18(day: pd.DataFrame) -> dict:
    """
    Brooks heuristic: by ~bar 18 the day has usually printed its high or low.
    We report what the extreme was at bar 18 and whether it held so far.
    """
    if len(day) < 18:
        return {"enough_bars": False}
    bnums = bar_number_series(day)
    cut = day[bnums <= 18]
    hi18 = float(cut["High"].max())
    lo18 = float(cut["Low"].min())
    # so-far extremes at the latest bar
    hi_now = float(day["High"].max())
    lo_now = float(day["Low"].min())
    hi_held = abs(hi18 - hi_now) < 1e-9  # still the day high
    lo_held = abs(lo18 - lo_now) < 1e-9  # still the day low
    return {
        "enough_bars": True,
        "high_at18": hi18, "low_at18": lo18,
        "high_still_day_high": hi_held,
        "low_still_day_low": lo_held
    }

def simple_bias(always_in: str, tr_score: float, or_status: str) -> str:
    """Combine Always-In, range score, and OR status into a single bias line."""
    if tr_score <= 0.25:
        ctx = "trend conditions"
    elif tr_score >= 0.75:
        ctx = "range conditions"
    else:
        ctx = "mixed conditions"
    if or_status == "above":
        or_clause = "trading above the opening range"
    elif or_status == "below":
        or_clause = "trading below the opening range"
    else:
        or_clause = "inside the opening range"
    return f"Bias: {always_in.upper()} under {ctx}, currently {or_clause}."

def recommend_setup(day: pd.DataFrame,
                    or_info: dict,
                    by18: dict,
                    mc_bull_last: int,
                    mc_bear_last: int,
                    always_in: str) -> dict:
    """
    Return {'label': ..., 'rationale': ...} or a 'no-setup' suggestion.
    Rules (Brooks-flavored heuristics):
      1) Trend pullback: strong Always-In + price just broke OR and pulls back near OR edge.
      2) Bar18 fade: by18 printed an extreme that is still holding; enter pullback toward that extreme.
      3) Failed breakout: price poked OR and re-entered → fade back to OR mid.
    """
    last_close = float(day["Close"].iloc[-1])
    # 1) Trend pullback off OR edge
    if always_in == "bull" and or_info["status"] == "above":
        if abs(last_close - or_info["high"]) <= 0.25 * (or_info["high"] - or_info["low"]):
            return {
                "label": "Buy pullback above OR high",
                "rationale": "Always-In bull with OR breakout; pulling back toward OR high acts as support."
            }
    if always_in == "bear" and or_info["status"] == "below":
        if abs(last_close - or_info["low"]) <= 0.25 * (or_info["high"] - or_info["low"]):
            return {
                "label": "Sell pullback below OR low",
                "rationale": "Always-In bear with OR breakout; pulling back toward OR low acts as resistance."
            }

    # 2) Bar-18 fade of the extreme that is still holding
    if by18.get("enough_bars"):
        rng = float(day["High"].max() - day["Low"].min())
        if rng > 0:
            # Near the low at 18 and low is still the day low → buy PB
            if by18.get("low_still_day_low") and abs(last_close - by18["low_at18"]) <= 0.2 * rng:
                return {
                    "label": "Buy pullback at morning low (Bar-18 hold)",
                    "rationale": "By Bar 18 the session often sets an extreme; low from Bar 18 still holds."
                }
            # Near the high at 18 and high is still the day high → sell PB
            if by18.get("high_still_day_high") and abs(last_close - by18["high_at18"]) <= 0.2 * rng:
                return {
                    "label": "Sell pullback at morning high (Bar-18 hold)",
                    "rationale": "By Bar 18 the session often sets an extreme; high from Bar 18 still holds."
                }

    # 3) Failed OR breakout re-entry
    # If price is back inside OR after being above/below earlier, fade to OR mid.
    last_status = or_info["status"]
    # reconstruct whether we were outside earlier: check any close > OR high or < OR low
    or_hi, or_lo = or_info["high"], or_info["low"]
    was_above = (day["Close"] > or_hi).any()
    was_below = (day["Close"] < or_lo).any()
    if last_status == "inside" and (was_above or was_below):
        return {
            "label": "Fade failed OR breakout toward OR mid",
            "rationale": "Breakout failed and price re-entered the range; mean reversion toward OR midpoint."
        }

    # Microchannel exhaustion hint
    if mc_bull_last >= 6:
        return {
            "label": "Wait for two-leg pullback after bull microchannel",
            "rationale": "Extended bull microchannel suggests exhaustion; higher-probability after two-leg PB."
        }
    if mc_bear_last >= 6:
        return {
            "label": "Wait for two-leg pullback after bear microchannel",
            "rationale": "Extended bear microchannel suggests exhaustion; higher-probability after two-leg PB."
        }

    return {"label": "No high-probability setup", "rationale": "Conditions lack clear edge under current rules."}

def narrative_text(or_info: dict, by18: dict) -> str:
    """Plain-English primer you described, with current session facts filled in."""
    lines = []
    lines.append("The market open sets the initial bias. The opening range is the first several bars (≈5–18).")
    lines.append("It marks the initial battle between bulls and bears; its high/low act as intraday S/R.")
    lines.append("")
    lines.append("Opening Range facts today:")
    lines.append(f"• Bars counted: {or_info['bars']}  • High: {or_info['high']:.2f}  • Low: {or_info['low']:.2f}")
    lines.append(f"• Price is currently {or_info['status']} the opening range.")
    lines.append("• Breakouts from this range often indicate a trend; failed breakouts often reverse.")
    if by18.get("enough_bars"):
        hi_hold = "still the day high" if by18["high_still_day_high"] else "not the current day high"
        lo_hold = "still the day low" if by18["low_still_day_low"] else "not the current day low"
        lines.append("")
        lines.append("Bar 18 heuristic:")
        lines.append(f"• High by Bar 18: {by18['high_at18']:.2f} ({hi_hold})")
        lines.append(f"• Low by Bar 18: {by18['low_at18']:.2f} ({lo_hold})")
        lines.append("• If price approaches that extreme and it holds, fading via a pullback can be reasonable.")
    return "\n".join(lines)


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

        # ---------- Session stats ----------
        o = float(day["Open"].iloc[0])
        c = float(day["Close"].iloc[-1])
        hi = float(day["High"].max())
        lo = float(day["Low"].min())
        pct_from_open = 100 * (c - o) / o
        pct_to_high = 100 * (hi - c) / c
        pct_to_low = 100 * (c - lo) / c

        # ---------- Core metrics ----------
        always, ema20, ema50 = always_in(day)
        mc_bull, mc_bear = microchannel_lengths(day)
        overlap = float(overlap_score(day, window=24).iloc[-1])
        bar18, b18d = bar18_flag(day)
        mm_up, mm_dn = measured_move(day)
        or_bo = opening_range_breakout(day)
        today_range, adr14 = range_vs_adr(day, df_all.set_index("Datetime"))

        # ---------- Derived context (MUST come before UI) ----------
        or_info = opening_range(day, bars_min=5, bars_max=18)
        by18 = hi_lo_by_bar18(day)
        bias_line = simple_bias(always, overlap, or_info["status"])
        setup = recommend_setup(
            day=day,
            or_info=or_info,
            by18=by18,
            mc_bull_last=int(mc_bull.iloc[-1]),
            mc_bear_last=int(mc_bear.iloc[-1]),
            always_in=always
        )

        # ---------- Flags ----------
        flags = []
        if always == "bull":
            flags.append("Always-In Bull")
        elif always == "bear":
            flags.append("Always-In Bear")
        if overlap >= 0.55:
            flags.append("Trading-Range Day")
        if max(int(mc_bull.iloc[-1]), int(mc_bear.iloc[-1])) >= 6:
            flags.append("Microchannel ≥6")
        if or_bo and or_bo.get("direction") == "up":
            flags.append("Opening BO Up")
        elif or_bo and or_bo.get("direction") == "down":
            flags.append("Opening BO Down")
        if bar18:
            flags.append("Bar-18 Exhaustion Risk")

        # ----------------------- Output -----------------------
        # Extra context
        hi_idx = day["High"].idxmax()
        lo_idx = day["Low"].idxmin()
        hi_time = day.loc[hi_idx, "Datetime"]
        lo_time = day.loc[lo_idx, "Datetime"]
        pos_in_range = (c - lo) / (hi - lo) if hi > lo else np.nan
        pos_pct = None if pd.isna(pos_in_range) else round(float(pos_in_range * 100), 2)

        # Top metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("% from Open", f"{pct_from_open:.2f}%")
        m2.metric("% to Session High", f"{pct_to_high:.2f}%")
        m3.metric("% to Session Low", f"{pct_to_low:.2f}%")
        m4.metric(
            "Day Range vs ADR14",
            f"{today_range:.2f} / {adr14:.2f}" if adr14 is not None else f"{today_range:.2f} / —"
        )

        n1, n2, n3 = st.columns(3)
        n1.metric("Always-In", always)
        n2.metric("Trading-Range Score (0–1)", f"{overlap:.2f}")
        n3.metric("Microchannel len (bull/bear)", f"{int(mc_bull.iloc[-1])}/{int(mc_bear.iloc[-1])}")

        # Trading plan summary
        st.subheader("Trading Plan Summary")
        st.markdown(f"**{bias_line}**")
        st.markdown(narrative_text(or_info, by18))
        st.markdown("**Setup**")
        st.markdown(f"- **{setup['label']}**")
        st.caption(setup["rationale"])

        # Flags
        st.subheader("Session Flags")
        st.write(", ".join(flags) if flags else "No special conditions flagged.")

        # Intraday context
        st.subheader("Intraday Context")
        ctx_rows = [
            ("Session High Time", hi_time.strftime("%H:%M")),
            ("Session Low Time", lo_time.strftime("%H:%M")),
            ("Position in Day Range", f"{pos_pct:.2f}%" if pos_pct is not None else "—"),
        ]
        if or_bo:
            dirn = or_bo.get("direction") or "none"
            lvl = or_bo.get("level")
            ft = or_bo.get("follow_through")
            ctx_rows += [
                ("Opening-Range Breakout", dirn.capitalize()),
                ("OR Level", f"{lvl:.2f}" if isinstance(lvl, (int, float)) else "—"),
                ("OR Follow-through", "Yes" if ft is True else ("No" if ft is False else "—")),
            ]
        mm_up_disp = None if mm_up is None else round(float(mm_up), 2)
        mm_dn_disp = None if mm_dn is None else round(float(mm_dn), 2)
        ctx_rows += [
            ("Measured Move Up", f"{mm_up_disp:.2f}" if mm_up_disp is not None else "—"),
            ("Measured Move Down", f"{mm_dn_disp:.2f}" if mm_dn_disp is not None else "—"),
        ]
        ctx_df = pd.DataFrame(ctx_rows, columns=["Metric", "Value"])
        st.table(ctx_df)

        # Bar-18 table
        st.subheader("Bar-18 Heuristic (bars 16–20)")
        b18 = {
            "bars_considered": b18d.get("bars_considered", "16–20"),
            "microchannel_len": b18d.get("microchannel_len", 0),
            "avg_body_pct": b18d.get("avg_body_pct", 0.0),
            "avg_stretch_vs_ema20": b18d.get("avg_stretch_vs_ema20", 0.0),
            "flag": bool(bar18),
        }
        b18_tbl = pd.DataFrame([
            ("Bars considered", b18["bars_considered"]),
            ("Microchannel len", str(int(b18["microchannel_len"]))),
            ("Avg body / bar range", f"{float(b18['avg_body_pct']):.3f}"),
            ("Avg stretch vs EMA20", f"{float(b18['avg_stretch_vs_ema20']):.4f}"),
            ("Exhaustion flag", "Yes" if b18["flag"] else "No"),
        ], columns=["Metric", "Value"])
        st.table(b18_tbl)

        # Bars table
        tbl = day.copy()
        tbl["ema20"] = ema20.values
        tbl["ema50"] = ema50.values
        tbl["bull_mc_len"] = mc_bull.values
        tbl["bear_mc_len"] = mc_bear.values
        st.subheader("Bars (latest 120)")
        st.dataframe(tbl.tail(120), use_container_width=True)

        # Plot
        st.subheader("Chart (Close with EMA20/50)")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(day["Datetime"], day["Close"], label="Close")
        ax.plot(day["Datetime"], ema20, label="EMA20")
        ax.plot(day["Datetime"], ema50, label="EMA50")
        ax.set_xlabel("Time"); ax.set_ylabel("Price"); ax.legend(loc="best")
        st.pyplot(fig)

        # Download
        st.download_button(
            "Download Day Bars CSV",
            day.to_csv(index=False).encode(),
            file_name=f"{symbol}_{last_date}_5m.csv",
            mime="text/csv"
        )
