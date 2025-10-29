
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Expected columns in df: Datetime, Open, High, Low, Close, Volume
# Datetime in ISO format or pandas-parseable
# Timeframe intended: 5-minute bars (works for 1m/15m with thresholds tweaked)

@dataclass
class BrooksContext:
    always_in: str            # 'bull', 'bear', or 'neutral'
    trading_range_score: float  # 0..1, higher = more rangebound
    volatility_ticks: float     # proxy: avg true range in ticks or percent
    ema_fast: float
    ema_slow: float

def _ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def _true_range(df):
    pc = df['Close'].shift(1)
    tr = pd.concat([(df['High'] - df['Low']).abs(),
                    (df['High'] - pc).abs(),
                    (df['Low'] - pc).abs()], axis=1).max(axis=1)
    return tr

def _bar_stats(df):
    rng = (df['High'] - df['Low']).replace(0, np.nan)
    body = (df['Close'] - df['Open'])
    body_abs = body.abs()
    body_pct = (body_abs / rng).fillna(0.0)
    upper_tail = (df['High'] - df[['Open','Close']].max(axis=1)) / rng
    lower_tail = (df[['Open','Close']].min(axis=1) - df['Low']) / rng
    bull = (body > 0).astype(int)
    bear = (body < 0).astype(int)
    is_trend_bar = (body_pct >= 0.6)  # closes in top/bottom 40% of range
    return pd.DataFrame({
        'rng': rng,
        'body': body,
        'body_pct': body_pct.clip(0,1),
        'upper_tail': upper_tail.clip(0,1).fillna(0.0),
        'lower_tail': lower_tail.clip(0,1).fillna(0.0),
        'bull': bull,
        'bear': bear,
        'is_trend_bar': is_trend_bar
    }, index=df.index)

def _always_in_direction(df, stats, ema_fast, ema_slow):
    # Heuristic for Brooks Always-In:
    # Direction flips on strong breakout with follow-through and position relative to slow EMA
    direction = []
    curr = 'neutral'
    for i in range(len(df)):
        if i < 5:
            direction.append(curr)
            continue
        close = df['Close'].iloc[i]
        prev_close = df['Close'].iloc[i-1]
        fast = ema_fast.iloc[i]
        slow = ema_slow.iloc[i]
        trendbar = stats['is_trend_bar'].iloc[i]
        # breakout conditions
        broke_high = close > df['High'].iloc[i-1] and prev_close > df['High'].iloc[i-2]
        broke_low  = close < df['Low'].iloc[i-1]  and prev_close < df['Low'].iloc[i-2]

        if trendbar and close > slow and (broke_high or curr == 'neutral'):
            curr = 'bull'
        if trendbar and close < slow and (broke_low or curr == 'neutral'):
            curr = 'bear'
        # soft persistence
        if curr == 'bull' and close < slow and not trendbar:
            curr = 'neutral'
        if curr == 'bear' and close > slow and not trendbar:
            curr = 'neutral'
        direction.append(curr)
    return pd.Series(direction, index=df.index)

def _microchannel_flags(df):
    # Bull microchannel: sequence of bars where lows are above prior lows
    # Bear microchannel: highs below prior highs
    low_above_prior = df['Low'] > df['Low'].shift(1)
    high_below_prior = df['High'] < df['High'].shift(1)
    bull_mc_len = []
    bear_mc_len = []
    b_run, s_run = 0, 0
    for i in range(len(df)):
        if low_above_prior.iloc[i]:
            b_run += 1
        else:
            b_run = 0
        if high_below_prior.iloc[i]:
            s_run += 1
        else:
            s_run = 0
        bull_mc_len.append(b_run)
        bear_mc_len.append(s_run)
    return pd.Series(bull_mc_len, index=df.index), pd.Series(bear_mc_len, index=df.index)

def _swings(df, lookback=1):
    highs = df['High']
    lows = df['Low']
    swing_high = (highs.shift(lookback) < highs.shift(lookback+1)) & (highs.shift(1) < highs)
    swing_low  = (lows.shift(lookback) > lows.shift(lookback+1)) & (lows.shift(1) > lows)
    return swing_high.fillna(False), swing_low.fillna(False)

def _count_pullbacks(signals, direction):
    seq = []
    c = 0
    for flag in signals:
        if flag:
            c += 1
            seq.append(c)
        else:
            seq.append(0)
    s = pd.Series(seq)
    if direction == 'bull':
        return s.replace(0, np.nan).ffill().fillna(0).astype(int).clip(0, 2)
    if direction == 'bear':
        return s.replace(0, np.nan).ffill().fillna(0).astype(int).clip(0, 2)
    return s

def _wedge_detector(pivots_series, prices, min_separation=5):
    idxs = list(pivots_series[pivots_series].index)
    if len(idxs) < 3:
        return pd.Series(False, index=prices.index)
    flags = pd.Series(False, index=prices.index)
    for i in range(2, len(idxs)):
        a, b, c = idxs[i-2], idxs[i-1], idxs[i]
        if (prices.index.get_loc(b) - prices.index.get_loc(a) >= min_separation and
            prices.index.get_loc(c) - prices.index.get_loc(b) >= min_separation):
            if prices.loc[a] < prices.loc[b] and prices.loc[b] < prices.loc[c]:
                flags.loc[c] = True
            if prices.loc[a] > prices.loc[b] and prices.loc[b] > prices.loc[c]:
                flags.loc[c] = True
    return flags

def _trading_range_score(df, window=30):
    hh = df['High'].rolling(window).max()
    ll = df['Low'].rolling(window).min()
    width = (hh - ll).replace(0, np.nan)
    mid = (hh + ll) / 2.0
    reentry = ((df['Close'] - mid).abs() < 0.2 * width).astype(int).rolling(window).mean()
    failed_bo = (((df['Close'] > hh.shift(1)) & (df['Close'].shift(1) < hh.shift(2))) |
                 ((df['Close'] < ll.shift(1)) & (df['Close'].shift(1) > ll.shift(2)))).rolling(window).mean()
    score = (0.7 * reentry.fillna(0) + 0.3 * failed_bo.fillna(0)).clip(0,1)
    return score

def _breakout_quality(df, stats, window=30):
    bh = df['Close'] > df['High'].shift(1)
    bl = df['Close'] < df['Low'].shift(1)
    ft_up = ((df['Close'].shift(-1) > df['Close']) & bh).rolling(window).mean()
    ft_dn = ((df['Close'].shift(-1) < df['Close']) & bl).rolling(window).mean()
    q = pd.Series(np.nan, index=df.index)
    q[bh] = ft_up[bh]
    q[bl] = ft_dn[bl]
    return q.fillna(method='ffill').fillna(0.5).clip(0,1)

def analyze_session(df: pd.DataFrame):
    df = df.copy()
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.set_index('Datetime').sort_index()
    df = df[['Open','High','Low','Close','Volume']].dropna()

    stats = _bar_stats(df)
    df['TR'] = (df['High'] - df['Low']).abs().combine(df['High'] - df['Close'].shift(1), max).combine(df['Low'] - df['Close'].shift(1), max)
    df['ATR14'] = df['TR'].rolling(14).mean()
    df['EMA9'] = _ema(df['Close'], 9)
    df['EMA20'] = _ema(df['Close'], 20)
    df['EMA50'] = _ema(df['Close'], 50)

    always_in_series = _always_in_direction(df, stats, df['EMA9'], df['EMA50'])
    tr_score = _trading_range_score(df, window=30)
    vol_pct = (df['ATR14'] / df['Close']).fillna(0.0)

    bull_mc, bear_mc = _microchannel_flags(df)

    sh, sl = _swings(df, lookback=1)
    h_seq = _count_pullbacks(sl.values, 'bull')
    b_seq = _count_pullbacks(sh.values, 'bear')
    H1 = (h_seq == 1)
    H2 = (h_seq >= 2)
    B1 = (b_seq == 1)
    B2 = (b_seq >= 2)

    wedge_high = _wedge_detector(sh, df['High'])
    wedge_low  = _wedge_detector(sl, df['Low'])

    bo_quality = _breakout_quality(df, stats, window=30)

    bull_signal = (stats['bull'].astype(bool) & (stats['lower_tail'] <= 0.25) & (stats['body_pct'] >= 0.5))
    bear_signal = (stats['bear'].astype(bool) & (stats['upper_tail'] <= 0.25) & (stats['body_pct'] >= 0.5))

    ctx = BrooksContext(
        always_in=always_in_series.iloc[-1],
        trading_range_score=float(tr_score.iloc[-1]),
        volatility_ticks=float(vol_pct.iloc[-1]),
        ema_fast=float(df['EMA20'].iloc[-1]),
        ema_slow=float(df['EMA50'].iloc[-1]),
    )

    last_sh_idx = df.index[sh].max() if sh.any() else None
    last_sl_idx = df.index[sl].max() if sl.any() else None
    mm_up = np.nan
    mm_dn = np.nan
    if last_sl_idx is not None:
        leg = df.loc[last_sl_idx:]['Close'].iloc[0] - df['Low'].loc[:last_sl_idx].min()
        mm_up = df['Close'].iloc[-1] + leg
    if last_sh_idx is not None:
        leg = df['High'].loc[:last_sh_idx].max() - df.loc[last_sh_idx:]['Close'].iloc[0]
        mm_dn = df['Close'].iloc[-1] - leg

    sig = pd.DataFrame(index=df.index)
    sig['always_in'] = always_in_series
    sig['bull_microchannel_len'] = bull_mc
    sig['bear_microchannel_len'] = bear_mc
    sig['H1'] = H1
    sig['H2'] = H2
    sig['B1'] = B1
    sig['B2'] = B2
    sig['wedge_high'] = wedge_high
    sig['wedge_low']  = wedge_low
    sig['bull_signal_bar'] = bull_signal
    sig['bear_signal_bar'] = bear_signal
    sig['bo_quality'] = bo_quality
    sig['trading_range_score'] = tr_score
    sig['atr_pct'] = vol_pct
    sig['ema20'] = df['EMA20']
    sig['ema50'] = df['EMA50']

    last = sig.iloc[-1]
    weights = {
        'trend': 0.25,
        'microchannel': 0.15,
        'pullback': 0.15,
        'breakout': 0.20,
        'volatility': 0.10,
        'wedge': 0.05,
        'range_penalty': 0.10
    }
    trend_score = 1.0 if ctx.always_in in ['bull','bear'] else 0.3
    trend_score *= 1.0 if (df['Close'].iloc[-1] > last['ema50'] if ctx.always_in=='bull' else df['Close'].iloc[-1] < last['ema50']) else 0.7

    micro_score = min(1.0, max(last['bull_microchannel_len'], last['bear_microchannel_len']) / 6.0)

    pullback_score = 0.0
    if ctx.always_in == 'bull':
        pullback_score = 1.0 if last['H2'] else 0.4 if last['H1'] else 0.2
    elif ctx.always_in == 'bear':
        pullback_score = 1.0 if last['B2'] else 0.4 if last['B1'] else 0.2

    breakout_score = float(last['bo_quality'])

    v = float(last['atr_pct'])
    if v <= 0:
        vol_score = 0.0
    elif v < 0.004:
        vol_score = v / 0.004
    elif v <= 0.02:
        vol_score = 1.0
    else:
        vol_score = max(0.1, 0.02 / v)

    wedge_score = 1.0 if (last['wedge_high'] or last['wedge_low']) else 0.3

    range_penalty = float(last['trading_range_score'])

    tradability_plus = (
        weights['trend'] * trend_score +
        weights['microchannel'] * micro_score +
        weights['pullback'] * pullback_score +
        weights['breakout'] * breakout_score +
        weights['volatility'] * vol_score +
        weights['wedge'] * wedge_score -
        weights['range_penalty'] * range_penalty
    )
    tradability_plus = round(100 * max(0.0, min(1.0, tradability_plus)), 2)

    return sig, ctx, {'mm_up': float(mm_up) if not np.isnan(mm_up) else None,
                      'mm_dn': float(mm_dn) if not np.isnan(mm_dn) else None,
                      'tradability_plus': tradability_plus}

def load_csv(path_or_buffer):
    df = pd.read_csv(path_or_buffer)
    cols_map = {c.lower(): c for c in df.columns}
    def pick(names):
        for n in names:
            if n in cols_map:
                return cols_map[n]
        return None
    dt = pick(['datetime','date','time','timestamp'])
    o = pick(['open'])
    h = pick(['high'])
    l = pick(['low'])
    c = pick(['close'])
    v = pick(['volume','vol'])
    req = [dt,o,h,l,c,v]
    if any(x is None for x in req):
        raise ValueError("CSV must include columns: Datetime, Open, High, Low, Close, Volume")
    df = df.rename(columns={dt:'Datetime', o:'Open', h:'High', l:'Low', c:'Close', v:'Volume'})
    return df[['Datetime','Open','High','Low','Close','Volume']]
