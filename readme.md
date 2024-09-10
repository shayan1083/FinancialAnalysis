A project to determine how a stock will move based on social media posts about it

Plan:

1) Gather Data
- Gather financial data, for now I will use msft
- Gather headlines and social media data, registered for reddit api and im searching r/stocks 
2) Preprocess Data
- Preprocess time series financial data
- clean and tokenize the text of headlines for sentiment analysis
3) Combine Features
- use time series analysis methods like adding lagged values or moving averages, will determine specifics later
- aggregate sentiment scores for each day, and use this as an additional input for time series
4) Build Hybrid Model
- Use a model to handle time series data, will determine specifics later
- incorporate sentiment score as an additional inpit feature with the time series data. The model should consider both historical and sentiment data to make predictions
5) Predict Stock Movement
- The model should have binary output, for increase or decrease. Later I can try to determine the specific increase and decrease amount
6) Evaluate
- evaluate the model

