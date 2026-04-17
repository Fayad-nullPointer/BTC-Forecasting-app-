import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from nixtla import NixtlaClient
import warnings
import os

warnings.filterwarnings('ignore')

st.set_page_config(page_title="BTC Forecast", layout="wide")

st.title("Bitcoin (BTC) Price Forecast")

# Sidebar Configuration
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload BTC Historical Data (CSV)", type=["csv"])

model_choice = st.sidebar.selectbox("Select Model", ["Prophet", "ARIMA", "ML Hybrid (ElasticNet + RF)", "Nixtla TimeGPT"])

api_key_input = ""
if model_choice == "Nixtla TimeGPT":
    default_key = ""
    if os.path.exists("API_key.txt"):
        try:
            with open("API_key.txt", "r") as f:
                default_key = f.read().strip()
        except:
            pass
    api_key_input = st.sidebar.text_input("Nixtla API Key", value=default_key, type="password")

forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=7, max_value=90, value=30, step=1)

# Nixtla officially supports 50, 80, 90 for their open examples, but can handle others. 
confidence_level = st.sidebar.selectbox("Confidence Interval (%)", [80, 90, 95], index=2)

st.sidebar.subheader("Technical Indicators")
show_sma = st.sidebar.checkbox("Show 30-Day SMA", value=False)
show_ema = st.sidebar.checkbox("Show 30-Day EMA", value=False)

generate_btn = st.sidebar.button("Generate Forecast")

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def engineer_features_ml(df):
    """Create features used in the notebook's ML Hybrid model."""
    d = df.copy()
    d['day_of_week'] = d['ds'].dt.dayofweek
    d['month'] = d['ds'].dt.month
    d['day_of_year'] = d['ds'].dt.dayofyear
    d['is_weekend'] = d['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Lags & Rolling
    d['lag_1'] = d['y'].shift(1)
    d['lag_3'] = d['y'].shift(3)
    d['lag_7'] = d['y'].shift(7)
    d['rolling_mean_7'] = d['y'].rolling(window=7).mean()
    d['rolling_std_7'] = d['y'].rolling(window=7).std()
    
    # Sinusoidal
    d["dayofweek_sinsuidal"] = np.sin(2 * np.pi * d["day_of_week"] / 7)
    d["month_sinsuidal"] = np.sin(2 * np.pi * d["month"] / 12)
    
    return d

def iterative_ml_forecast(train_df, horizon):
    """Iteratively predict future values to simulate a multi-step horizon."""
    hist = train_df.copy()
    preds = []
    
    for _ in range(horizon):
        feats = engineer_features_ml(hist)
        # Drop NaNs just for training
        train_feats = feats.dropna()
        
        X_train = train_feats.drop(columns=['ds', 'y'])
        y_train = train_feats['y']
        
        # 1. Fit ElasticNet (Trend)
        trend_model = ElasticNetCV(cv=5)
        trend_model.fit(X_train, y_train)
        
        # 2. Detrend
        y_train_detrend = y_train - trend_model.predict(X_train)
        
        # 3. Fit RF on Residuals
        rf = RandomForestRegressor(bootstrap=False, random_state=42)
        rf.fit(X_train, y_train_detrend)
        
        # 4. Predict Next Step
        next_date = hist['ds'].iloc[-1] + pd.Timedelta(days=1)
        next_row = pd.DataFrame({'ds': [next_date], 'y': [hist['y'].iloc[-1]]}) # dummy y
        
        temp = pd.concat([hist, next_row], ignore_index=True)
        temp_feats = engineer_features_ml(temp)
        
        X_next = temp_feats.drop(columns=['ds', 'y']).iloc[[-1]]
        
        p_trend = trend_model.predict(X_next)[0]
        p_res = rf.predict(X_next)[0]
        p_final = p_trend + p_res
        
        preds.append(p_final)
        
        # Add the prediction to history for the next iteration
        hist = pd.concat([hist, pd.DataFrame({'ds': [next_date], 'y': [p_final]})], ignore_index=True)
        
    return np.array(preds)


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
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
            price_col = next((c for c in df.columns if 'close' in c.lower()), df.columns[1])

        df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
        df = df.sort_values(by=date_col)
        df_model = df[[date_col, price_col]].rename(columns={date_col: 'ds', price_col: 'y'})
        df_model.dropna(inplace=True)

        st.subheader("Historical Data Preview")
        st.dataframe(df.tail())

        if generate_btn:
            if model_choice == "Nixtla TimeGPT" and not api_key_input:
                st.error("Nixtla API Key is required for the TimeGPT model.")
                st.stop()
                
            with st.spinner(f"Training {model_choice} model and forecasting {forecast_horizon} days..."):
                # Backtesting split
                train_size = len(df_model) - forecast_horizon
                if train_size < 30:
                    st.error("Not enough data for the selected forecast horizon.")
                    st.stop()
                    
                train = df_model.iloc[:train_size]
                test = df_model.iloc[train_size:]
                
                forecast_dates = pd.date_range(start=df_model['ds'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
                
                # Base Figure
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_model['ds'], y=df_model['y'], mode='lines', name='Historical BTC Price', line=dict(color='blue')))
                
                if show_sma:
                    sma = df_model['y'].rolling(window=30).mean()
                    fig.add_trace(go.Scatter(x=df_model['ds'], y=sma, mode='lines', name='30-Day SMA', line=dict(color='orange')))
                if show_ema:
                    ema = df_model['y'].ewm(span=30, adjust=False).mean()
                    fig.add_trace(go.Scatter(x=df_model['ds'], y=ema, mode='lines', name='30-Day EMA', line=dict(color='green')))

                y_pred, y_lower, y_upper, f_dates = None, None, None, None
                preds_bt = None

                if model_choice == "Prophet":
                    # Backtest
                    m_bt = Prophet(interval_width=confidence_level/100.0, daily_seasonality=True)
                    m_bt.fit(train)
                    preds_bt = m_bt.predict(m_bt.make_future_dataframe(periods=forecast_horizon))['yhat'].iloc[-forecast_horizon:].values
                    
                    # Full Forecast
                    m = Prophet(interval_width=confidence_level/100.0, daily_seasonality=True)
                    m.fit(df_model)
                    forecast_future = m.predict(m.make_future_dataframe(periods=forecast_horizon)).iloc[-forecast_horizon:]
                    
                    y_pred = forecast_future['yhat'].values
                    y_lower = forecast_future['yhat_lower'].values
                    y_upper = forecast_future['yhat_upper'].values
                    f_dates = forecast_future['ds']
                    
                elif model_choice == "ARIMA":
                    # Backtest
                    y_train = train['y'].values
                    model_bt = auto_arima(y_train, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
                    preds_bt = model_bt.predict(n_periods=forecast_horizon)
                    
                    # Full Forecast
                    model = auto_arima(df_model['y'].values, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
                    preds, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True, alpha=1-(confidence_level/100.0))
                    
                    y_pred = preds
                    y_lower = conf_int[:, 0]
                    y_upper = conf_int[:, 1]
                    f_dates = forecast_dates
                    
                elif model_choice == "Nixtla TimeGPT":
                    nixtla_client = NixtlaClient(api_key=api_key_input)
                    
                    # Backtest
                    fcst_bt = nixtla_client.forecast(
                        df=train, h=forecast_horizon, target_col='y', time_col='ds', level=[confidence_level]
                    )
                    preds_bt = fcst_bt['TimeGPT'].values
                    
                    # Full Forecast
                    fcst = nixtla_client.forecast(
                        df=df_model, h=forecast_horizon, target_col='y', time_col='ds', level=[confidence_level]
                    )
                    y_pred = fcst['TimeGPT'].values
                    y_lower = fcst[f'TimeGPT-lo-{confidence_level}'].values
                    y_upper = fcst[f'TimeGPT-hi-{confidence_level}'].values
                    f_dates = fcst['ds']
                    
                elif model_choice == "ML Hybrid (ElasticNet + RF)":
                    # Backtest
                    preds_bt = iterative_ml_forecast(train, forecast_horizon)
                    
                    # Full Forecast
                    y_pred = iterative_ml_forecast(df_model, forecast_horizon)
                    
                    # Simple empirical bounds for the ML Hybrid as tree/linear combos don't do native prediction intervals easily
                    error_margin = (100 - confidence_level) / 100.0
                    y_lower = y_pred * (1 - error_margin)
                    y_upper = y_pred * (1 + error_margin)
                    f_dates = forecast_dates

                # Calculate metrics
                mae, rmse = compute_metrics(test['y'].values, preds_bt)

                # Convert to arrays to safely plot
                f_dates_arr = np.array(f_dates)
                
                # Add Projected Trend
                fig.add_trace(go.Scatter(
                    x=f_dates_arr, 
                    y=y_pred, 
                    mode='lines', 
                    name=f'Projected Trend ({model_choice})', 
                    line=dict(color='red', dash='dash')
                ))

                # Add Uncertainty Zone
                fig.add_trace(go.Scatter(
                    x=np.concatenate([f_dates_arr, f_dates_arr[::-1]]),
                    y=np.concatenate([y_upper, y_lower[::-1]]),
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{confidence_level}% Confidence Interval'
                ))
                
                # Add a marker for the start of the forecast
                fig.add_trace(go.Scatter(
                    x=[f_dates_arr[0]],
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
        st.error(f"Error processing the request: {e}")
else:
    st.info("Please upload a CSV file with BTC historical data to begin.")
