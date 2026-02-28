import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Trend Advisor", layout="wide")

# --- UI Controls ---
st.title("📈 AI Stock Trend Lab")
ticker = st.sidebar.text_input("Stock Ticker", value="NVDA").upper()
days = st.sidebar.slider("Analysis Days", 5, 200, 60)
chart_style = st.sidebar.selectbox("Chart Style", ["Candlestick", "Bar Chart", "Line Chart"])

# --- Logic ---
try:
    data = yf.download(ticker, period="1y", interval="1d", progress=False)
    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        df = data.tail(days).copy()
        df['Day_Index'] = np.arange(len(df))
        
        # Regression
        model = LinearRegression().fit(df[['Day_Index']].values, df['Close'].values.reshape(-1, 1))
        pred = model.predict([[len(df)]])[0][0].item()
        
        # FIX: Ensure metrics are clean scalars
        st.metric("Forecast Tomorrow", f"₹{pred:.2f}")

        # --- Plotting (The FIX for image_1efb84.png) ---
        fig = go.Figure()
        if chart_style == "Candlestick":
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines'))

        # Trendline - using CORRECT method add_trace
        trend_pts = model.predict(np.arange(len(df)).reshape(-1, 1)).flatten()
        fig.add_trace(go.Scatter(x=df.index, y=trend_pts, name="Lion Guard Trend", line=dict(color='orange', dash='dot')))

        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error: {e}")
