A project to learn about time series analysis and some machine learning applications for financial data.

Possible Applications
- Predict daily returns
- Forecast volatility
- Predict direction of stock movement
- Relative performance



Plan:

1) Gather Data
- Gather financial data, for now I will use msft
- Gather headlines and social media data, registered for reddit api and will use praw to get data
2) Preprocess Data
- use time series analysis methods like adding lagged values or moving averages, will determine specifics later
- aggregate sentiment scores for each day, and use this as an additional input for time series
3) Build Model
- Use a model to handle time series data. I starting with simple models and adding more complex ones afterwards
- Naive -> Exponential Smoothing -> Winter-Holts -> ARIMA -> GARCH -> ARIMA-GARCH -> Boosted Trees -> RNN -> LSTM -> TFT
- incorporate sentiment score as an additional input feature with the time series data. The model should consider both historical and sentiment data to make predictions
4) Predict Stock Movement
- The model should output the daily return
5) Evaluate
- evaluate the model

