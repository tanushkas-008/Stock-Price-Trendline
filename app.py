import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# 1. FIXED: Correct Plotly method is 'add_trace', not 'add'
# 2. FIXED: Closed all string literals to avoid SyntaxError

st.title("📈 Stock Trend Advisor")

ticker = st.text_input("Enter Ticker", value="NVDA").upper()
days = st.slider("Lookback Days", 5, 100, 60)

try:
    data = yf.download(ticker, period="1y", interval="1d", progress=False)
    if not data.empty:
        # Handle MultiIndex columns if necessary
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        df = data.tail(days).copy()
        df['Day_Index'] = np.arange(len(df))
        
        # Regression Logic
        model = LinearRegression().fit(df[['Day_Index']].values, df['Close'].values.reshape(-1, 1))
        pred = model.predict([[len(df)]])[0][0]
        
        # Create Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close Price"))
        
        # Trendline Logic
        trend_pts = model.predict(np.arange(len(df)).reshape(-1, 1)).flatten()
        # FIXED: Use 'add_trace' as shown in your error log
        fig.add_trace(go.Scatter(
            x=df.index, y=trend_pts, 
            name='Lion Guard Trend',
            line=dict(color='orange', width=2, dash='dot')
        ))
        
        fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # FIXED: Ensure all print/markdown strings are closed properly
        st.write(f"Next Day Predicted Price: {pred:.2f}")

except Exception as e:
    st.error(f"Error: {e}")
