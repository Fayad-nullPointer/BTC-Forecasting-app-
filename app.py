import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
from pmdarima import auto_arima
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="BTC Forecast", layout="wide")

st.title("Bitcoin (BTC) Price Forecast")

# Sidebar Configuration
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload BTC Historical Data (CSV)", type=["csv"])

model_choice = st.sidebar.selectbox("Select Model", ["Prophet", "ARIMA"])
forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=30, step=1)
confidence_level = st.sidebar.selectbox("Confidence Interval", [80, 95], index=1)

st.sidebar.subheader("Technical Indicators")
show_sma = st.sidebar.checkbox("Show 30-Day SMA", value=False)
show_ema = st.sidebar.checkbox("Show 30-Day EMA", value=False)

generate_btn = st.sidebar.button("Generate Forecast")

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

if uploaded_file is not None:
    # Load Data
    try:
        df = pd.read_csv(uploaded_file)
        # Ensure we have date and close price columns. Assuming 'Close time' and 'Close' as in the notebook.
        if "Close time" in df.columns:
            date_col = "Close time"
        elif "Date" in df.columns:
            date_col = "Date"
        elif "timestamp" in df.columns:
            date_col = "timestamp"
        else:
            date_col = df.columns[0]
            
        if "Close" in df.columns:
            price_col = "Close"
        else:
            # Try to find a column with 'close' in it
            price_col = next((c for c in df.columns if 'close' in c.lower()), df.columns[1])

        df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
        df = df.sort_values(by=date_col)
        df_model = df[[date_col, price_col]].rename(columns={date_col: 'ds', price_col: 'y'})
        df_model.dropna(inplace=True)

        st.subheader("Historical Data Preview")
        st.dataframe(df.tail())

        if generate_btn:
            with st.spinner(f"Training {model_choice} model and forecasting {forecast_horizon} days..."):
                # Splitting data for backtesting (last row - horizon)
                train_size = len(df_model) - forecast_horizon
                if train_size < 30:
                    st.error("Not enough data for the selected forecast horizon.")
                    st.stop()
                    
                train = df_model.iloc[:train_size]
                test = df_model.iloc[train_size:]

                forecast_dates = pd.date_range(start=df_model['ds'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
                
                # Plotly Figure
                fig = go.Figure()

                # Add historical data
                fig.add_trace(go.Scatter(x=df_model['ds'], y=df_model['y'], mode='lines', name='Historical BTC Price', line=dict(color='blue')))
                
                if show_sma:
                    sma = df_model['y'].rolling(window=30).mean()
                    fig.add_trace(go.Scatter(x=df_model['ds'], y=sma, mode='lines', name='30-Day SMA', line=dict(color='orange')))
                if show_ema:
                    ema = df_model['y'].ewm(span=30, adjust=False).mean()
                    fig.add_trace(go.Scatter(x=df_model['ds'], y=ema, mode='lines', name='30-Day EMA', line=dict(color='green')))

                if model_choice == "Prophet":
                    # Prophet Model
                    m = Prophet(interval_width=confidence_level/100.0, daily_seasonality=True)
                    m.fit(df_model)
                    
                    # Backtest
                    m_bt = Prophet()
                    m_bt.fit(train)
                    future_bt = m_bt.make_future_dataframe(periods=forecast_horizon)
                    forecast_bt = m_bt.predict(future_bt)
                    preds_bt = forecast_bt['yhat'].iloc[-forecast_horizon:].values
                    
                    # Full Forecast
                    future = m.make_future_dataframe(periods=forecast_horizon)
                    forecast = m.predict(future)
                    
                    forecast_future = forecast.iloc[-forecast_horizon:]
                    
                    y_pred = forecast_future['yhat']
                    y_lower = forecast_future['yhat_lower']
                    y_upper = forecast_future['yhat_upper']
                    f_dates = forecast_future['ds']
                    
                elif model_choice == "ARIMA":
                    # ARIMA Model using auto_arima
                    y_train = train['y'].values
                    model_bt = auto_arima(y_train, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
                    preds_bt, conf_int_bt = model_bt.predict(n_periods=forecast_horizon, return_conf_int=True, alpha=1-(confidence_level/100.0))
                    
                    model = auto_arima(df_model['y'].values, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
                    preds, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True, alpha=1-(confidence_level/100.0))
                    
                    y_pred = preds
                    y_lower = conf_int[:, 0]
                    y_upper = conf_int[:, 1]
                    f_dates = forecast_dates

                # Calculate metrics
                mae, rmse = compute_metrics(test['y'].values, preds_bt)

                # Add Projected Trend
                fig.add_trace(go.Scatter(
                    x=f_dates, 
                    y=y_pred, 
                    mode='lines', 
                    name='Projected Trend', 
                    line=dict(color='red', dash='dash')
                ))

                # Add Uncertainty Zone
                fig.add_trace(go.Scatter(
                    x=pd.concat([f_dates, f_dates[::-1]]),
                    y=pd.concat([pd.Series(y_upper), pd.Series(y_lower)[::-1]]),
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{confidence_level}% Confidence Interval'
                ))
                
                # Add a marker for the start of the forecast
                fig.add_trace(go.Scatter(
                    x=[f_dates.iloc[0]],
                    y=[df_model['y'].iloc[-1]],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='star'),
                    name='Forecast Start'
                ))

                fig.update_layout(
                    title=f'BTC Price Forecast ({model_choice} - {forecast_horizon} Days)',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    hovermode='x unified',
                    template='plotly_white'
                )

                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Backtesting Performance Metrics")
                st.info(f"Tested on the last {forecast_horizon} days of uploaded data.")
                col1, col2 = st.columns(2)
                col1.metric("MAE (Mean Absolute Error)", f"${mae:,.2f}")
                col2.metric("RMSE (Root Mean Square Error)", f"${rmse:,.2f}")

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("Please upload a CSV file with BTC historical data to begin.")
