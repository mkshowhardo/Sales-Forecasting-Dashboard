# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.arima.model import ARIMAResults

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.title("📊 Sales Forecasting Dashboard")
st.markdown("""
Welcome! This app forecasts future sales using **ARIMA** / **SARIMA** models.  
Upload your sales CSV file with **Date** and **Sales** columns or use the default dataset.
""")

# ========================
# Load default dataset
# ========================
try:
    df = pd.read_csv("dataset/train.csv", parse_dates=["Date"])
except FileNotFoundError:
    st.warning("Default dataset not found. Please upload your CSV file.")
    df = None

df['Date'] = pd.to_datetime(df['Date'])
df.rename(columns={'Weekly_Sales': 'Sales'}, inplace=True)

#========================
# File uploader
# ========================
uploaded_file = st.file_uploader("Upload your CSV file (Date, Sales)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    st.success("File uploaded successfully!")

# ========================
# Show dataset
# ========================
if df is not None:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Plot sales trend
    st.subheader("Sales Trend")
    plt.figure(figsize=(12,5))
    plt.plot(df['Date'], df['Sales'], marker='o')
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Sales Over Time")
    st.pyplot(plt)

    # ========================
    # Load Models
    # ========================
    try:
        arima_model = joblib.load("model/arima_model.pkl")
        sarima_model = joblib.load("model/sarimax_model.pkl")
        st.success("Models loaded successfully!")
    except FileNotFoundError:
        st.warning("Model files not found. Make sure they are in the 'model/' folder.")
        arima_model = None
        sarima_model = None

    # ========================
    # Forecast
    # ========================
    if arima_model is not None and sarima_model is not None:
        st.subheader("Forecast Options")
        months = st.slider("Months to Forecast", min_value=1, max_value=36, value=12)

        forecast_type = st.radio("Choose model", ["ARIMA", "SARIMA"])

        if st.button("Generate Forecast"):
            if forecast_type == "ARIMA":
                forecast = arima_model.get_forecast(steps=months)
                pred_mean = forecast.predicted_mean
                conf_int = forecast.conf_int()
            else:
                forecast = sarima_model.get_forecast(steps=months)
                pred_mean = forecast.predicted_mean
                conf_int = forecast.conf_int()

            # Plot forecast
            plt.figure(figsize=(12,5))
            plt.plot(df['Date'], df['Sales'], label="Historical")
            future_dates = pd.date_range(df['Date'].iloc[-1], periods=months+1, freq='M')[1:]
            plt.plot(future_dates, pred_mean, label="Forecast", color='orange')
            plt.fill_between(future_dates, conf_int.iloc[:,0], conf_int.iloc[:,1], color='orange', alpha=0.2)
            plt.xlabel("Date")
            plt.ylabel("Sales")
            plt.title(f"{forecast_type} Forecast for {months} Months")
            plt.legend()
            st.pyplot(plt)

            st.success("Forecast generated successfully!")
else:
    st.info("Upload a CSV or place the default dataset in 'dataset/train.csv'.")

st.markdown("---")
st.markdown("Made by MK SHOWHARDO")