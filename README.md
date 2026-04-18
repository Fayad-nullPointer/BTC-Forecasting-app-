<div align="center">
  <img src="https://cryptologos.cc/logos/bitcoin-btc-logo.png" alt="Bitcoin Logo" width="100"/>
  <h1>Crypto Forecaster - BTC 🪙</h1>
  <p><b>Advanced Time-Series Analysis and Machine Learning Forecasting for Bitcoin</b></p>
  <p><i>Developed by Ahmed Fayad</i></p>
</div>

---

## 📖 Introduction
The **Crypto Forecaster** is a robust, interactive web application built with Streamlit. It is designed to forecast the future prices of Bitcoin (BTC) utilizing various advanced time-series analysis and machine learning models. The project bridges the gap between traditional statistical methods and modern foundational AI approaches to provide accurate, customizable forecasts.

## ✨ Key Features
- **Interactive UI**: Built using Streamlit for seamless historical CSV data uploads.
- **Dynamic Configuration**: Adjust forecast horizons (from 7 up to 90 days) and specific confidence intervals (80%, 90%, 95%).
- **Technical Indicators**: Toggle 30-Day Simple Moving Averages (SMA) and Exponential Moving Averages (EMA).
- **Rich Visualizations**: Interactive Plotly charts showing historical data, forecasted trends, and shaded uncertainty zones.
- **Tabular Projections**: Clear date-to-price forecasted dataframe rendering with upper and lower bounds.

## 🚀 Quick Start / Installation

### Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 1. Clone the repository
```bash
git clone https://github.com/your-username/crypto-forecaster-btc.git
cd crypto-forecaster-btc
```

### 2. Install dependencies
Install the required packages using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
*(Dependencies include: `streamlit`, `pandas`, `numpy`, `plotly`, `prophet`, `pmdarima`, `scikit-learn`, `nixtla`)*

### 3. Set up Nixtla API (Optional but recommended)
To use the TimeGPT model, you need a Nixtla API key. 
Create a file named `API_key.txt` in the root directory and paste your API key inside.
```bash
echo "YOUR_NIXTLA_API_KEY" > API_key.txt
```

### 4. Run the application
```bash
streamlit run app.py
```

## 🧠 Forecasting Models & Technical Architecture

The application implements four distinct predictive models to extract patterns from highly volatile time-series data:

### 1. Prophet
Designed by Meta, Prophet frames forecasting as a curve-fitting additive model dealing with trend, seasonality, and holidays.
- **Strength**: Highly robust to missing data, shifts in the trend, and large outliers.
- **Limitation**: Treats forecasting purely as a curve-fitting problem. Struggles with unprecedented market shocks (black swan events) unless explicitly configured with external regressors.

### 2. ARIMA (Auto-Regressive Integrated Moving Average)
A classical statistical model utilizing the `pmdarima` library to automatically find optimal parameters ($p, d, q$) for stationarity, autoregression, and moving averages.
- **Strength**: Excellent baseline model for understanding long-term statistical trends and stability.
- **Limitation**: Acts as a smoothing linear model; it struggles to capture the complex, non-linear micro-movements and unpredictable variance inherently tied to crypto assets.

### 3. Nixtla TimeGPT
The first generative pre-trained transformer model purposefully built for time-series forecasting.
- **Strength**: Uses multi-head self-attention mechanisms and transfer learning (pre-trained on billions of global time-series data points) for highly accurate zero-shot inference without local fine-tuning.
- **Limitation**: Functions as a "black box" deep learning model via API, which introduces external dependency and reduces explicit interpretability for financial analysts.

### 4. ML Hybrid Model (ElasticNet + Random Forest)
A custom-built pipeline to overcome the inability of tree-based models to extrapolate trends.
- **Phase 1 (ElasticNetCV)**: A regularized linear regression model extracts the core directional trend.
- **Phase 2 (Random Forest)**: A Random Forest Regressor is trained exclusively on the detrended residuals to capture complex, high-frequency market noise.
- **Iterative Auto-Regression**: Sequentially calculates $t+1$ to engineer rolling means and lags for future predictions.
- **Limitation**: Error propagation. Small errors in early step predictions can cascade and magnify as the forecast horizon extends.

## 🛠 Technologies Used
- **Frontend / Framework**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning / AI**: Scikit-Learn, Prophet, pmdarima, Nixtla
- **Data Visualization**: Plotly

## 📄 License & Copyright
© Developed by Ahmed Fayad  
Information Technology Institute (ITI)
