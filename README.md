# BTC-Forecasting-App

This project focuses on forecasting Bitcoin (BTC) prices using daily historical data (2018-2025). The core of the project involves exploring various time-series forecasting methodologies in a Jupyter Notebook, which are then deployed into an interactive Streamlit application.

## Models Explored

During the experimental phase (`notebook.ipynb`), four distinct modeling approaches were evaluated to handle the extreme volatility and non-linear trends inherent to cryptocurrency markets:

### 1. ARIMA (AutoRegressive Integrated Moving Average)
- **Approach:** A foundational statistical method for time-series forecasting.
- **Implementation:** Utilized `pmdarima`'s `auto_arima` to search for optimal parameters. 
- **Techniques:** To satisfy ARIMA's stationarity requirements, the data required meticulous preprocessing, specifically log transformations (to stabilize variance) and differencing (to stabilize the mean). 
- **Insight:** While providing a mathematical baseline, direct application of ARIMA struggled to capture the complex, non-linear volatility of Bitcoin over long horizons.

### 2. Facebook Prophet
- **Approach:** A regression model that works exceptionally well with strong seasonal effects.
- **Implementation:** Configured with multiplicative seasonality to adapt to rapid macroeconomic trend shifts. 
- **Techniques:** Added custom Fourier-based monthly seasonality (period=30, fourier order=10) and tuned change-point priors. The model was further enhanced by injecting shifted trading volume (`volume_lag1`) as an external regressor, significantly improving its precision.

### 3. Nixtla TimeGPT
- **Approach:** A generative pretrained foundation model specifically designed for time series forecasting.
- **Implementation:** Leveraged the Nixtla API for zero-shot forecasting.
- **Techniques:** Applied log-transformed variables and used `finetune_steps=1`. 
- **Insight:** TimeGPT showed impressive out-of-the-box performance, structuring highly confident prediction intervals around future volatility without requiring the exhaustive feature engineering needed by traditional ML models.

### 4. Machine Learning Hybrid (ElasticNet + Random Forest)
- **Approach:** A custom hybrid pipeline designed to solve the "Extrapolation Problem" inherent to tree-based models (which cannot predict values outside their absolute training domain boundaries).
- **Implementation:** 
  - **Feature Engineering:** Extracted temporal features (lag_1, lag_3, lag_7, rolling means/stds) and cyclical sinusoidal embeddings (for day and month).
  - **Trend Capture:** An `ElasticNet` model was first trained to capture and project the overarching linear trend.
  - **Residual Learning:** A `RandomForestRegressor` was then trained on the detrended residuals (the true price minus the ElasticNet trend prediction).
- **Insight:** By summing the ElasticNet trend prediction and the Random Forest residual prediction, this hybrid approach successfully provided accurate, trend-aware, and highly responsive deterministic predictions.

## Application Interface
The optimal parameters from above have been integrated into `app.py`. The Streamlit app provides a complete UI allowing users to upload data, choose a model, tweak confidence intervals, and trace future BTC price trends effortlessly.
