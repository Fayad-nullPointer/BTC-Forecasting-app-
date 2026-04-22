import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from prophet import Prophet
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from nixtla import NixtlaClient
import warnings
import os

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Crypto Forecaster", page_icon="🪙", layout="wide")

# Add Logo and Project Name
col1, col2 = st.columns([1, 15])
with col1:
    st.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=60)
with col2:
    st.title("Crypto Forecaster - BTC")

st.markdown("<p><b><i>© Developed by Ahmed Fayad</i></b></p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload BTC Historical Data (CSV)", type=["csv"])

model_choice = st.sidebar.selectbox("Select Model", ["Prophet", "ARIMA", "ML Hybrid (ElasticNet + RF)", "Nixtla TimeGPT"])

price_value = st.sidebar.selectbox("Select Price Value", ["Close", "Open", "High", "Low"])

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

confidence_level = st.sidebar.selectbox("Confidence Interval (%)", [80, 90, 95], index=2)

st.sidebar.subheader("Technical Indicators")
show_sma = st.sidebar.checkbox("Show 30-Day SMA", value=False)
show_ema = st.sidebar.checkbox("Show 30-Day EMA", value=False)

generate_btn = st.sidebar.button("Generate Forecast")

def engineer_features_ml(df):
    """Create features used in the notebook's ML Hybrid model."""
    d = df.copy()
    d['day_of_week'] = d['ds'].dt.dayofweek
    d['month'] = d['ds'].dt.month
    d['day_of_year'] = d['ds'].dt.dayofyear
    d['is_weekend'] = d['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    d['lag_1'] = d['y'].shift(1)
    d['lag_3'] = d['y'].shift(3)
    d['lag_7'] = d['y'].shift(7)
    d['rolling_mean_7'] = d['y'].rolling(window=7).mean()
    d['rolling_std_7'] = d['y'].rolling(window=7).std()
    
    d["dayofweek_sinsuidal"] = np.sin(2 * np.pi * d["day_of_week"] / 7)
    d["month_sinsuidal"] = np.sin(2 * np.pi * d["month"] / 12)
    
    return d

def iterative_ml_forecast(train_df, horizon):
    """Iteratively predict future values."""
    hist = train_df.copy()
    preds = []
    
    for _ in range(horizon):
        feats = engineer_features_ml(hist)
        train_feats = feats.dropna()
        
        X_train = train_feats.drop(columns=['ds', 'y'])
        y_train = train_feats['y']
        
        trend_model = ElasticNetCV(cv=5)
        trend_model.fit(X_train, y_train)
        y_train_detrend = y_train - trend_model.predict(X_train)
        
        rf = RandomForestRegressor(bootstrap=False, random_state=42)
        rf.fit(X_train, y_train_detrend)
        
        next_date = hist['ds'].iloc[-1] + pd.Timedelta(days=1)
        next_row = pd.DataFrame({'ds': [next_date], 'y': [hist['y'].iloc[-1]]})
        
        temp = pd.concat([hist, next_row], ignore_index=True)
        temp_feats = engineer_features_ml(temp)
        
        X_next = temp_feats.drop(columns=['ds', 'y']).iloc[[-1]]
        
        p_trend = trend_model.predict(X_next)[0]
        p_res = rf.predict(X_next)[0]
        p_final = p_trend + p_res
        
        preds.append(p_final)
        hist = pd.concat([hist, pd.DataFrame({'ds': [next_date], 'y': [p_final]})], ignore_index=True)
        
    return np.array(preds)


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())
        
        if "Close time" in df.columns:
            date_col = "Close time"
        elif "Date" in df.columns:
            date_col = "Date"
        elif "timestamp" in df.columns:
            date_col = "timestamp"
        else:
            date_col = df.columns[0]
            
        # Find the requested price column
        possible_cols = [c for c in df.columns if price_value.lower() == c.lower().strip()]
        if not possible_cols:
            possible_cols = [c for c in df.columns if price_value.lower() in c.lower() and 'time' not in c.lower() and 'date' not in c.lower()]
            
        if possible_cols:
            price_col = possible_cols[0]
        else:
            st.warning(f"Could not find exact '{price_value}' column. Using an alternative.")
            price_col = df.columns[1]

        df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)
        df = df.sort_values(by=date_col)
        df_model = df[[date_col, price_col]].rename(columns={date_col: 'ds', price_col: 'y'})
        df_model['y'] = pd.to_numeric(df_model['y'], errors='coerce')
        df_model.dropna(inplace=True)

        if generate_btn:
            if model_choice == "Nixtla TimeGPT" and not api_key_input:
                st.error("Nixtla API Key is required for the TimeGPT model.")
                st.stop()
                
            if model_choice == "Nixtla TimeGPT":
                nixtla_client = NixtlaClient(api_key=api_key_input)
                
            with st.spinner(f"Training {model_choice} model, calculating error metrics, and forecasting {forecast_horizon} days..."):
                
                # Evaluation (Train/Test Split on Historical Data)
                train_eval = df_model.iloc[:-forecast_horizon]
                test_eval = df_model.iloc[-forecast_horizon:]
                if len(train_eval) > 50:
                    y_eval_pred = None
                    if model_choice == "Prophet":
                        m_eval = Prophet(daily_seasonality=True)
                        m_eval.fit(train_eval)
                        f_eval = m_eval.make_future_dataframe(periods=forecast_horizon)
                        y_eval_pred = m_eval.predict(f_eval).iloc[-forecast_horizon:]['yhat'].values
                    elif model_choice == "ARIMA":
                        m_eval = auto_arima(train_eval['y'].values, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
                        y_eval_pred = m_eval.predict(n_periods=forecast_horizon)
                    elif model_choice == "Nixtla TimeGPT":
                        fcst_eval = nixtla_client.forecast(
                            df=train_eval, h=forecast_horizon, target_col='y', time_col='ds'
                        )
                        y_eval_pred = fcst_eval['TimeGPT'].values
                    elif model_choice == "ML Hybrid (ElasticNet + RF)":
                        y_eval_pred = iterative_ml_forecast(train_eval, forecast_horizon)
                        
                    y_true = test_eval['y'].values
                    mae = mean_absolute_error(y_true, y_eval_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_eval_pred))
                    mape = np.mean(np.abs((y_true - y_eval_pred) / y_true)) * 100
                    
                    st.subheader(f"Model Error Metrics (Last {forecast_horizon} Days Backtest)")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("MAE", f"${mae:,.2f}")
                    col_m2.metric("RMSE", f"${rmse:,.2f}")
                    col_m3.metric("MAPE", f"{mape:.2f}%")
                    st.markdown("---")
                    
                # Full Forecast
                forecast_dates = pd.date_range(start=df_model['ds'].iloc[-1] + pd.Timedelta(days=1), periods=forecast_horizon)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_model['ds'], y=df_model['y'], mode='lines', name='Historical BTC Price', line=dict(color='blue')))
                
                if show_sma:
                    sma = df_model['y'].rolling(window=30).mean()
                    fig.add_trace(go.Scatter(x=df_model['ds'], y=sma, mode='lines', name='30-Day SMA', line=dict(color='orange')))
                if show_ema:
                    ema = df_model['y'].ewm(span=30, adjust=False).mean()
                    fig.add_trace(go.Scatter(x=df_model['ds'], y=ema, mode='lines', name='30-Day EMA', line=dict(color='green')))

                y_pred, y_lower, y_upper, f_dates = None, None, None, None

                if model_choice == "Prophet":
                    m = Prophet(interval_width=confidence_level/100.0, daily_seasonality=True)
                    m.fit(df_model)
                    forecast_future = m.predict(m.make_future_dataframe(periods=forecast_horizon)).iloc[-forecast_horizon:]
                    
                    y_pred = forecast_future['yhat'].values
                    y_lower = forecast_future['yhat_lower'].values
                    y_upper = forecast_future['yhat_upper'].values
                    f_dates = forecast_future['ds']
                    
                elif model_choice == "ARIMA":
                    model = auto_arima(df_model['y'].values, seasonal=False, trace=False, error_action='ignore', suppress_warnings=True)
                    preds, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True, alpha=1-(confidence_level/100.0))
                    
                    y_pred = preds
                    y_lower = conf_int[:, 0]
                    y_upper = conf_int[:, 1]
                    f_dates = forecast_dates
                    
                elif model_choice == "Nixtla TimeGPT":
                    nixtla_client = NixtlaClient(api_key=api_key_input)
                    fcst = nixtla_client.forecast(
                        df=df_model, h=forecast_horizon, target_col='y', time_col='ds', level=[confidence_level]
                    )
                    y_pred = fcst['TimeGPT'].values
                    y_lower = fcst[f'TimeGPT-lo-{confidence_level}'].values
                    y_upper = fcst[f'TimeGPT-hi-{confidence_level}'].values
                    f_dates = fcst['ds']
                    
                elif model_choice == "ML Hybrid (ElasticNet + RF)":
                    y_pred = iterative_ml_forecast(df_model, forecast_horizon)
                    
                    error_margin = (100 - confidence_level) / 100.0
                    y_lower = y_pred * (1 - error_margin)
                    y_upper = y_pred * (1 + error_margin)
                    f_dates = forecast_dates

                f_dates_arr = np.array(f_dates)
                
                # Made projected forecasted line more visible by increasing line width and changing style
                fig.add_trace(go.Scatter(
                    x=f_dates_arr, 
                    y=y_pred, 
                    mode='lines+markers', 
                    name=f'Projected Trend ({model_choice})', 
                    line=dict(color='red', width=4),
                    marker=dict(size=6, color='darkred')
                ))

                fig.add_trace(go.Scatter(
                    x=np.concatenate([f_dates_arr, f_dates_arr[::-1]]),
                    y=np.concatenate([y_upper, y_lower[::-1]]),
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{confidence_level}% Confidence Interval'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[f_dates_arr[0]],
                    y=[df_model['y'].iloc[-1]],
                    mode='markers',
                    marker=dict(color='green', size=12, symbol='star'),
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

                st.subheader(f"Forecasted Values for the Next {forecast_horizon} Days")
                
                # Prepare forecasted dataframe for display
                forecast_df = pd.DataFrame({
                    "Date": pd.to_datetime(f_dates_arr).strftime('%Y-%m-%d'),
                    "Forecasted Price (USD)": [f"${val:,.2f}" for val in y_pred],
                    "Lower Bound": [f"${val:,.2f}" for val in y_lower],
                    "Upper Bound": [f"${val:,.2f}" for val in y_upper]
                })
                
                # Show dataframe to user
                st.dataframe(forecast_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing the request: {e}")
else:
    st.info("Please upload a CSV file with BTC historical data to begin.")
