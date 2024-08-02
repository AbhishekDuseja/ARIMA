import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Define a function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Define a function to generate ARIMA model and forecast
def arima_forecast(data, order, steps):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return model_fit, forecast

# Streamlit app
st.title('Stock ARIMA Forecasting Web App')

st.write("""
This app allows you to input a stock ticker and fit an ARIMA model to forecast future stock prices.
""")

# User inputs
ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, MSFT):', 'AAPL')
start_date = st.date_input('Start Date:', pd.to_datetime('2023-01-01'))
end_date = st.date_input('End Date:', pd.to_datetime('2024-01-01'))

# Ensure end_date is after start_date
if end_date <= start_date:
    st.error("End date must be after start date.")
else:
    st.write(f"Fetching data for {ticker} from {start_date} to {end_date}...")

    # Fetch and display stock data
    data = fetch_stock_data(ticker, start_date, end_date)
    
    if data.empty:
        st.error(f"No data found for ticker {ticker}.")
    else:
        st.write("Stock Data Preview:")
        st.write(data.head())

        st.write("Stock Price Plot with Moving Averages:")
        
        # Calculate moving averages
        data['MA100'] = data['Close'].rolling(window=100).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()

        # Plot stock price with moving averages
        plt.figure(figsize=(14, 7))
        plt.plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.6)
        plt.plot(data.index, data['MA100'], label='MA100', color='orange')
        plt.plot(data.index, data['MA200'], label='MA200', color='green')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{ticker} Price and Moving Averages')
        plt.legend()
        st.pyplot(plt)

        # Display ACF and PACF plots for diagnostics
        st.write("Autocorrelation and Partial Autocorrelation Plots:")
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(data['Close'].dropna(), ax=ax[0])
        plot_pacf(data['Close'].dropna(), ax=ax[1])
        st.pyplot(fig)

        # Fixed ARIMA parameters
        p = 5
        d = 1
        q = 5
        steps = 15

        st.write(f"Using ARIMA parameters: p={p}, d={d}, q={q} and forecasting {steps} steps.")

        # Fit ARIMA model and forecast
        if st.button('Forecast'):
            with st.spinner('Fitting ARIMA model...'):
                model_fit, forecast = arima_forecast(data['Close'], (p, d, q), steps)

                st.write("Forecast:")
                st.write(pd.DataFrame(forecast, columns=['Forecast']))

                # Plot forecast
                plt.figure(figsize=(10, 5))
                plt.plot(data.index, data['Close'], label='Observed', color='blue')
                
                # Generate forecast index manually
                last_date = data.index[-1]
                forecast_index = [last_date + pd.DateOffset(days=x) for x in range(1, steps + 1)]
                
                plt.plot(forecast_index, forecast, label='Forecast', color='orange')
                plt.xlabel('Date')
                plt.ylabel('Close Price')
                plt.title(f'ARIMA Forecast for {ticker}')
                plt.legend()
                st.pyplot(plt)
