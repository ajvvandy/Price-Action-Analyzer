
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from brooks_price_action import load_csv, analyze_session

st.set_page_config(page_title="Brooks Price Action Analyzer", layout="wide")
st.title("Brooks Price Action Analyzer")

st.write("Upload 5-minute OHLCV CSV to compute Al-Brooks style context and signals. Educational use only.")

uploaded = st.file_uploader("CSV file (columns: Datetime, Open, High, Low, Close, Volume)", type=["csv"])

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
