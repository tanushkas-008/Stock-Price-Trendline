import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import timedelta

st.set_page_config(page_title="Pro Stock Predictor", layout="wide")

# --- STEP 1: SMART SEARCH ENGINE ---
def get_ticker_from_name(name):
    try:
        # This searches Yahoo Finance for the best match based on your input
        search = yf.Search(name, max_results=1)
        if search.quotes:
            return search.quotes[0]['symbol']
    except:
        return None
    return None

# --- SIDEBAR INPUTS ---
st.sidebar.header("🕹️ Control Panel")

# 1. Name-based Input (Like TradingView)
user_input = st.sidebar.text_input("Search Stock Name (e.g. Nvidia, Google, Tata Motors)", value="Nvidia")

# 2. Prediction Days
days_to_predict = st.sidebar.slider("Historical Analysis Days", 5, 200, 60)

# 3. Chart Type Ticker (Dropdown)
chart_style = st.sidebar.selectbox("Select Chart View", ["Candlestick", "Bar Chart", "Line Chart"])

# --- PROCESSING ---
ticker = get_ticker_from_name(user_input)

if ticker:
    st.subheader(f"📊 Analysis for: {user_input} ({ticker})")
    
    # Fetch Data
    data = yf.download(ticker, period="1y", interval="1d", progress=False)
    
    if not data.empty:
        # Fix for yfinance MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        df = data.tail(days_to_predict).copy()
        df['Day_Index'] = np.arange(len(df))
        
        # ML Logic (Linear Regression)
        X = df[['Day_Index']].values
        y = df['Close'].values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        
        # Predict Tomorrow
        prediction = model.predict([[len(df)]])[0][0].item()
        current_price = df['Close'].iloc[-1].item()
        slope = model.coef_[0][0].item()

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Live Price", f"{current_price:.2f}")
        c2.metric("Target Tomorrow", f"{prediction:.2f}", f"{prediction-current_price:+.2f}")
        c3.metric("Trend", "UP 🚀" if slope > 0 else "DOWN 🔻")

        # --- DYNAMIC CHARTING (The Fix for 'add_trace') ---
        fig = go.Figure()

        if chart_style == "Candlestick":
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Market"))
        elif chart_style == "Bar Chart":
            fig.add_trace(go.Bar(x=df.index, y=df['Close'], name="Daily Close"))
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines+markers', name="Trend Line"))

        # Add ML Trendline
        trend_vals = model.predict(np.arange(len(df)).reshape(-1, 1)).flatten()
        fig.add_trace(go.Scatter(x=df.index, y=trend_vals, name="ML Trendline", line=dict(color='orange', dash='dot')))

        # Target Star
        next_date = df.index[-1] + timedelta(days=1)
        fig.add_trace(go.Scatter(x=[next_date], y=[prediction], mode='markers', name='Forecast', marker=dict(color='cyan', size=15, symbol='star')))

        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No data found for this stock.")
else:
    st.warning("Could not find a ticker for that name. Please be more specific (e.g. 'Apple Inc').")
