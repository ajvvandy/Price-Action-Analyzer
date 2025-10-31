from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from streamlit.components.v1 import html as st_html

st.set_page_config(page_title="Brooks Price-Action Day Report", layout="centered")

st.markdown("""
<style>
/* Global font + colors (very strong selectors) */
html, body, [data-testid="stAppViewContainer"], .stApp, .stMarkdown, .stText, .stPlotlyChart, .stAltairChart, .stMetric,
[data-testid="stMarkdownContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] * {
    font-family: 'Times New Roman', serif !important;
    color: #000000 !important;
}
.stApp { background-color: #ffffff !important; }

/* Container width & centering */
.block-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Times New Roman', serif !important;
    color: #000000 !important;
    font-weight: 600 !important;
    text-align: center !important;
    margin: 0.5rem 0 0.75rem 0 !important;
}

/* Paragraph & lists */
p, li, div, span {
    font-size: 1.05rem !important;
    line-height: 1.55rem !important;
}

/* Cards (flat/minimal) */
.card { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0.5rem 0; }

/* Metrics */
[data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: 500 !important; }

/* Inputs */
input, textarea {
    font-family: 'Times New Roman', serif !important;
    color: #000000 !important;
}

/* Analyze button: black background, white text */
.stButton button, .analyze-btn button, div.stButton > button {
    background-color: #000000 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 0px !important;
    font-family: 'Times New Roman', serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    padding: 0.6rem 1.2rem !important;
}
.stButton button:hover, .analyze-btn button:hover, div.stButton > button:hover {
    background-color: #222222 !important;
}

/* Tables */
table { border-collapse: collapse !important; margin-top: 1rem !important; }
th, td { border: none !important; padding: 0.4rem 0.6rem !important; text-align: left !important; }

/* Hide Streamlit default header/footer chrome for minimal look */
header, footer { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)






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
def tradingview_widget(symbol: str, theme: str = "dark", height: int = 560) -> None:
    # AAPL -> NASDAQ:AAPL; MSFT -> NASDAQ:MSFT; fallback to uppercase only
    tv_symbol = f"NASDAQ:{symbol.upper()}" if symbol.upper() not in ("SPY","QQQ","DIA") else symbol.upper()
    st_html(f"""
    <div class="tradingview-widget-container" style="height:{height}px;">
      <div id="tradingview_chart" style="height:{height}px;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
          "width": "100%",
          "height": {height},
          "symbol": "{tv_symbol}",
          "interval": "5",
          "timezone": "America/New_York",
          "theme": "{theme}",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "rgba(0,0,0,0)",
          "enable_publishing": false,
          "hide_side_toolbar": false,
          "allow_symbol_change": true,
          "studies": ["MASimple@tv-basicstudies","MAExp@tv-basicstudies"],
          "container_id": "tradingview_chart"
      }});
      </script>
    </div>
    """, height=height)

def bar_number_series(df: pd.DataFrame) -> pd.Series:
    """1-based bar numbers for the session."""
    return pd.Series(np.arange(1, len(df)+1), index=df.index, name="bar")

def day_outlook_prediction(overlap: float,
                           always_in: str,
                           by18: dict,
                           or_info: dict,
                           or_bo: Optional[dict]) -> dict:
    """
    Returns {'label': 'Bullish/Range/Bearish', 'bull': p, 'range': p, 'bear': p}
    Heuristic:
      - overlap ↑ ⇒ range probability ↑
      - Always-In tilts trend side
      - Bar-18 extreme holds tilt toward side opposite the held extreme being broken
      - OR breakout tilts trend side (if present)
    """
    # Start with range probability from overlap, clamp to [0,1]
    range_prob = float(np.clip(overlap, 0.0, 1.0))
    trend_pool = 1.0 - range_prob

    bull_prob = 0.5 * trend_pool
    bear_prob = 0.5 * trend_pool

    # Always-In tilt
    if always_in == "bull":
        bull_prob += 0.15 * trend_pool
        bear_prob -= 0.15 * trend_pool
    elif always_in == "bear":
        bear_prob += 0.15 * trend_pool
        bull_prob -= 0.15 * trend_pool

    # OR breakout tilt
    if or_bo:
        if or_bo.get("direction") == "up":
            bull_prob += 0.10 * trend_pool
            bear_prob -= 0.10 * trend_pool
        elif or_bo.get("direction") == "down":
            bear_prob += 0.10 * trend_pool
            bull_prob -= 0.10 * trend_pool

    # Bar-18 heuristic tilts
    if by18.get("enough_bars"):
        # If morning high is still the day high → bearish tilt
        if by18.get("high_still_day_high"):
            bear_prob += 0.12 * trend_pool
            bull_prob -= 0.12 * trend_pool
        # If morning low is still the day low → bullish tilt
        if by18.get("low_still_day_low"):
            bull_prob += 0.12 * trend_pool
            bear_prob -= 0.12 * trend_pool

    # Normalize
    bull_prob = max(0.0, bull_prob)
    bear_prob = max(0.0, bear_prob)
    range_prob = max(0.0, range_prob)
    s = bull_prob + bear_prob + range_prob
    if s <= 1e-9:
        bull_prob = bear_prob = 0.0
        range_prob = 1.0
        s = 1.0
    bull_prob /= s; bear_prob /= s; range_prob /= s

    # Label
    probs = {"Bullish": bull_prob, "Range": range_prob, "Bearish": bear_prob}
    label = max(probs, key=probs.get)
    return {"label": label, "bull": bull_prob, "range": range_prob, "bear": bear_prob}


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
        or_clause = " above the opening range"
    elif or_status == "below":
        or_clause = " below the opening range"
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
    """Facts-only summary. No explanations."""
    lines = []
    lines.append("**Opening Range facts today:**")
    lines.append(f"• Bars counted: {or_info['bars']}")
    lines.append(f"• High: {or_info['high']:.2f}")
    lines.append(f"• Low: {or_info['low']:.2f}")
    lines.append(f"• Price status: {or_info['status']}")
    if by18.get("enough_bars"):
        hi_hold = "still the day high" if by18["high_still_day_high"] else "not the current day high"
        lo_hold = "still the day low" if by18["low_still_day_low"] else "not the current day low"
        lines.append("")
        lines.append("**Bar 18:**")
        lines.append(f"• High by Bar 18: {by18['high_at18']:.2f} ({hi_hold})")
        lines.append(f"• Low by Bar 18: {by18['low_at18']:.2f} ({lo_hold})")
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
        bar18, b18d = bar18_flag(day)  # or bar18_flag(day, start_bar=..., end_bar=...)
        mm_up, mm_dn = measured_move(day)
        or_bo = opening_range_breakout(day)
        today_range, adr14 = range_vs_adr(day, df_all.set_index("Datetime"))

        # ---------- Derived context (MUST come before UI) ----------
        or_info = opening_range(day, bars_min=5, bars_max=18)  # or your slider value
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

        # ---------- Top metrics ----------
        n1, n2, n3 = st.columns(3)
        n1.metric("Always-In", always.upper())
        n2.metric("Day Outlook", outlook["label"],
                  delta=f"Bull {outlook['bull']:.0%} • Range {outlook['range']:.0%} • Bear {outlook['bear']:.0%}")
        n3.metric("Microchannel len (bull/bear)", f"{int(mc_bull.iloc[-1])}/{int(mc_bear.iloc[-1])}")


        # ---------- Flags ----------
       # ---------- Flags (more specific) ----------
        flags = []

        # Always-In state
        if always == "bull":
            flags.append("Always-In: Bull")
        elif always == "bear":
            flags.append("Always-In: Bear")
        else:
            flags.append("Always-In: Neutral")

        # Trading-range intensity
        if overlap >= 0.75:
            flags.append("Range Bias: Strong")
        elif overlap >= 0.60:
            flags.append("Range Bias: Moderate")
        elif overlap <= 0.25:
            flags.append("Trend Bias: Strong")
        else:
            flags.append("Trend Bias: Mixed")

        # Opening-range breakout specifics
        if or_bo:
            if or_bo.get("direction") == "up":
                flags.append("OR Breakout: Up")
                flags.append(f"OR Follow-through: {'Yes' if or_bo.get('follow_through') else 'No'}")
            elif or_bo.get("direction") == "down":
                flags.append("OR Breakout: Down")
                flags.append(f"OR Follow-through: {'Yes' if or_bo.get('follow_through') else 'No'}")
            else:
                # Detect failed re-entry: previously outside, currently inside
                or_hi, or_lo = or_info["high"], or_info["low"]
                was_above = (day["Close"] > or_hi).any()
                was_below = (day["Close"] < or_lo).any()
                if or_info["status"] == "inside" and (was_above or was_below):
                    flags.append("Failed OR Breakout → Re-entry")

        # Bar-18 specifics (90% heuristic incorporated as labeling)
        if by18.get("enough_bars"):
            if by18["high_still_day_high"]:
                flags.append("Bar-18: Morning High is Day High (90% heuristic)")
            if by18["low_still_day_low"]:
                flags.append("Bar-18: Morning Low is Day Low (90% heuristic)")

        # Microchannel exhaustion hint
        if max(int(mc_bull.iloc[-1]), int(mc_bear.iloc[-1])) >= 6:
            side = "Bull" if int(mc_bull.iloc[-1]) >= int(mc_bear.iloc[-1]) else "Bear"
            flags.append(f"Microchannel ≥6: {side}-side exhaustion risk")


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

        # Always-In
        n1.metric("Always-In", always.upper())

        # Day Outlook prediction
        outlook = day_outlook_prediction(overlap, always, by18, or_info, or_bo)
        n2.metric("Day Outlook", outlook["label"],
                  delta=f"Bull {outlook['bull']:.0%} • Range {outlook['range']:.0%} • Bear {outlook['bear']:.0%}")

        # Microchannel summary
        n3.metric("Microchannel len (bull/bear)",
                  f"{int(mc_bull.iloc[-1])}/{int(mc_bear.iloc[-1])}")


        # Trading plan summary
        st.subheader("Trading Plan Summary")
        st.markdown(f"**{bias_line}**")
        st.markdown(narrative_text(or_info, by18))

        # Big, prominent Setup block
        st.markdown(
            f"""
        <div style="margin-top:0.5rem; padding:0.75rem 1rem; border-left:6px solid rgba(0,0,0,0.25); background:rgba(0,0,0,0.03); border-radius:8px;">
          <div style="font-size:1.25rem; font-weight:700; margin-bottom:0.25rem;">{setup['label']}</div>
          <div style="opacity:0.85; font-size:0.95rem;">{setup['rationale']}</div>
        </div>   
        """,
         unsafe_allow_html=True,
        )


        # Flags
        st.subheader("Session Flags")
        st.write(", ".join(flags) if flags else "No special conditions flagged.")

        
        # Plot
        st.subheader("Chart (Close with EMA20/50)")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(day["Datetime"], day["Close"], label="Close")
        ax.plot(day["Datetime"], ema20, label="EMA20")
        ax.plot(day["Datetime"], ema50, label="EMA50")
        ax.set_xlabel("Time"); ax.set_ylabel("Price"); ax.legend(loc="best")
        st.pyplot(fig)

       # Interactive chart (TradingView)
        st.subheader("Interactive Chart (TradingView)")
        tradingview_widget(symbol, theme="light")



        # Download
        st.download_button(
            "Download Day Bars CSV",
            day.to_csv(index=False).encode(),
            file_name=f"{symbol}_{last_date}_5m.csv",
            mime="text/csv"
        )
