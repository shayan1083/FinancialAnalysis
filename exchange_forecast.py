import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import yfinance as yf
import argparse

from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.stattools import adfuller

def open_sql_connection():
    load_dotenv()
    return create_engine(f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASS')}@localhost/stock_data")

def obtain_econ_data():
    """
    Obtains economic indicator data from a MySQL database and returns it as a pandas DataFrame.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the economic indicator data, with the record_date column as the index and the index converted to a monthly period.
    """
    engine = open_sql_connection()
    query = 'select * from economic_indicators'
    economic_indicators = pd.read_sql(query, engine)
    economic_indicators = economic_indicators.set_index('record_date')
    economic_indicators = economic_indicators.sort_index()
    economic_indicators.index = pd.DatetimeIndex(economic_indicators.index).to_period('M')
    return economic_indicators

def modify_economic_data(economic_indicators: pd.DataFrame):
    """
    Modifies the economic data obtained from the database by handling missing values, calculating ratios and differences between economic indicators, and returning the cleaned and transformed data.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the modified economic indicator data, with the record_date column as the index and the index converted to a monthly period.
    """
    economic_indicators['Euro Interest Rate'] = economic_indicators['Euro Interest Rate'].replace(0, np.nan)
    economic_indicators['Euro Interest Rate'] = economic_indicators['Euro Interest Rate'].ffill()
    economic_indicators['Euro Inflation Rate'] = economic_indicators['Euro Inflation Rate'].replace(0,np.nan)
    economic_indicators['Euro Inflation Rate'] = economic_indicators['Euro Inflation Rate'].ffill()

    economic_indicators['CCI_Ratio'] = (economic_indicators['Euro CCI']/economic_indicators['US CCI'])
    economic_indicators['Inflation_Ratio'] = (economic_indicators['Euro Inflation Rate']/economic_indicators['US Inflation Rate'])
    economic_indicators['GDP_Ratio'] = (economic_indicators['US_GDP_Growth']/economic_indicators['Euro_GDP_Growth'])
    economic_indicators['Interest_Ratio'] = (economic_indicators['US Interest Rate']/economic_indicators['Euro Interest Rate'])

    economic_indicators['CCI_Difference'] = economic_indicators['US CCI'] - economic_indicators['Euro CCI']
    economic_indicators['Inflation_Difference'] = economic_indicators['US Inflation Rate'] - economic_indicators['Euro Inflation Rate']
    economic_indicators['GDP_Difference'] = economic_indicators['US_GDP_Growth'] - economic_indicators['Euro_GDP_Growth']
    economic_indicators['Interest_Difference'] = economic_indicators['US Interest Rate'] - economic_indicators['Euro Interest Rate']
    economic_indicators = economic_indicators.dropna()
    return economic_indicators

def obtain_exchange_data():
    """
    Obtains the daily USD/EUR exchange rate data from Yahoo Finance, resamples it to a monthly frequency, and returns the resulting DataFrame.
    
    Args:
        len_econ_data (int): The length of the economic data, used to determine the start and end dates for the exchange rate data.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the monthly USD/EUR exchange rate.
    """
    usd_eur = yf.download('USDEUR=X', start='2014-01-01', end='2024-01-01', interval='1d', progress=False)
    usd_eur = usd_eur.resample('ME').mean()
    usd_eur.index = pd.DatetimeIndex(usd_eur.index).to_period('M')
    return usd_eur

def combine_both_dataframes(economic_indicators: pd.DataFrame, usd_eur: pd.DataFrame):
    """
    Combines the economic indicator data and the USD/EUR exchange rate data into a single DataFrame, ensuring that the data is aligned by the record_date index.
    
    Args:
        economic_indicators (pandas.DataFrame): A DataFrame containing the economic indicator data.
        usd_eur (pandas.DataFrame): A DataFrame containing the USD/EUR exchange rate data.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the combined economic indicator and USD/EUR exchange rate data, with the record_date index aligned.
    """
    cut = len(economic_indicators) - len(usd_eur)
    if cut > 0:
        economic_indicators.drop(economic_indicators.tail(cut).index, inplace=True)
    close = usd_eur['Close'].values
    new_close = pd.Series(close[:len(economic_indicators.index)], index=economic_indicators.index)
    final = economic_indicators
    final['Close'] = new_close
    final.index = pd.to_datetime(final.index.to_timestamp()).to_period('M').to_timestamp() 
    return final

def plot_us_indicators(df: pd.DataFrame):
    """
    Plots the US economic indicators (Consumer Confidence Index, Inflation Rate, GDP Growth, and Interest Rate) in a 2x2 grid of subplots.
    
    Args:
        df (pandas.DataFrame): A DataFrame containing the economic indicator data, including the columns 'US CCI', 'US Inflation Rate', 'US_GDP_Growth', and 'US Interest Rate'.
    
    Returns:
        None
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    num_ticks = 10  

    ax1.plot(df.index.astype(str), df['US CCI'], label = 'US CCI', color='b')
    ax1.set_title('US Consumer Confidence Index', fontsize=14)
    ax1.set_xlabel('Record Date', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend()
    ax1.set_xticks(ax1.get_xticks()[::len(df)//num_ticks])
    ax1.tick_params(axis='x', rotation=45)


    ax2.plot(df.index.astype(str), df['US Inflation Rate'], label='US Inflation Rate', color='r')
    ax2.set_title('US Inflation Rate', fontsize=14)
    ax2.set_xlabel('Record Date', fontsize=12)
    ax2.set_ylabel('Percent', fontsize=12)
    ax2.legend()
    ax2.set_xticks(ax2.get_xticks()[::len(df)//num_ticks])
    ax2.tick_params(axis='x', rotation=45)

    ax3.plot(df.index.astype(str), df['US_GDP_Growth'], label='US GDP Growth', color='g')
    ax3.set_title('US GDP Growth', fontsize=14)
    ax3.set_xlabel('Record Date', fontsize=12)
    ax3.set_ylabel('Percent', fontsize=12)
    ax3.legend()
    ax3.set_xticks(ax3.get_xticks()[::len(df)//num_ticks])
    ax3.tick_params(axis='x', rotation=45)


    ax4.plot(df.index.astype(str), df['US Interest Rate'], label='US Interest Rate', color='y')
    ax4.set_title('US Interest Rate', fontsize=14)
    ax4.set_xlabel('Record Date', fontsize=12)
    ax4.set_ylabel('Percent', fontsize=12)
    ax4.legend()
    ax4.set_xticks(ax4.get_xticks()[::len(df)//num_ticks])
    ax4.tick_params(axis='x', rotation=45)
    plt.subplots_adjust(hspace=1)
    plt.show(block=True)

def plot_eur_indicators(df: pd.DataFrame):
    """
    Plots the Euro economic indicators (Consumer Confidence Index, Inflation Rate, GDP Growth, and Interest Rate) in a 2x2 grid of subplots.
    
    Args:
        df (pandas.DataFrame): A DataFrame containing the economic indicator data, including the columns 'Euro CCI', 'Euro Inflation Rate', 'Euro_GDP_Growth', and 'Euro Interest Rate'.
    
    Returns:
        None
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    num_ticks = 10  

    ax1.plot(df.index.astype(str), df['Euro CCI'], label = 'Euro CCI', color='b')
    ax1.set_title('Euro Consumer Confidence Index', fontsize=14)
    ax1.set_xlabel('Record Date', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend()
    ax1.set_xticks(ax1.get_xticks()[::len(df)//num_ticks])
    ax1.tick_params(axis='x', rotation=45)


    ax2.plot(df.index.astype(str), df['Euro Inflation Rate'], label='Euro Inflation Rate', color='r')
    ax2.set_title('Euro Inflation Rate', fontsize=14)
    ax2.set_xlabel('Record Date', fontsize=12)
    ax2.set_ylabel('Percent', fontsize=12)
    ax2.legend()
    ax2.set_xticks(ax2.get_xticks()[::len(df)//num_ticks])
    ax2.tick_params(axis='x', rotation=45)

    ax3.plot(df.index.astype(str), df['Euro_GDP_Growth'], label='Euro GDP Growth', color='g')
    ax3.set_title('Euro GDP Growth', fontsize=14)
    ax3.set_xlabel('Record Date', fontsize=12)
    ax3.set_ylabel('Percent', fontsize=12)
    ax3.legend()
    ax3.set_xticks(ax3.get_xticks()[::len(df)//num_ticks])
    ax3.tick_params(axis='x', rotation=45)


    ax4.plot(df.index.astype(str), df['Euro Interest Rate'], label='Euro Interest Rate', color='y')
    ax4.set_title('Euro Interest Rate', fontsize=14)
    ax4.set_xlabel('Record Date', fontsize=12)
    ax4.set_ylabel('Percent', fontsize=12)
    ax4.legend()
    ax4.set_xticks(ax4.get_xticks()[::len(df)//num_ticks])
    ax4.tick_params(axis='x', rotation=45)

    plt.subplots_adjust(hspace=1)
    plt.show(block=True)

def plot_indicator_relationships(df: pd.DataFrame):
    """
    Plots the differences and ratios of various economic indicators (Consumer Confidence Index, Inflation Rate, GDP Growth, and Interest Rate) between the US and Euro regions over time.
    
    Args:
        df (pandas.DataFrame): A DataFrame containing the economic indicator data, including the columns 'CCI_Difference', 'Inflation_Difference', 'GDP_Difference', 'Interest_Difference', 'CCI_Ratio', 'Inflation_Ratio', 'GDP_Ratio', and 'Interest_Ratio'.
    
    Returns:
        None
    """
    fig2, ((ax1, ax2)) = plt.subplots(1,2,figsize=(12,6))
    num_ticks = 10 
    ax1.plot(df.index.astype(str), df['CCI_Difference'], label='CCI Difference', color='b')
    ax1.plot(df.index.astype(str), df['Inflation_Difference'], label='Inflation Difference', color='r')
    ax1.plot(df.index.astype(str), df['GDP_Difference'], label='GDP Growth Difference', color='g')
    ax1.plot(df.index.astype(str), df['Interest_Difference'], label='Interest Rate Difference', color='y')
    ax1.set_title('Various Differentials Over Time', fontsize=14)
    ax1.set_xlabel('Record Date', fontsize=12)
    ax1.set_ylabel('Difference', fontsize=12)
    ax1.legend()
    ax1.set_xticks(ax1.get_xticks()[::len(df)//num_ticks])
    ax1.tick_params(axis='x', rotation=45)

    ax2.plot(df.index.astype(str), df['CCI_Ratio'], label='CCI Ratio', color='b')
    ax2.plot(df.index.astype(str), df['Inflation_Ratio'], label='Inflation Ratio', color='r')
    ax2.plot(df.index.astype(str), df['GDP_Ratio'], label='GDP Growth Ratio', color='g')
    ax2.plot(df.index.astype(str), df['Interest_Ratio'], label='Interest Ratio', color='y')
    ax2.set_title('Various Ratios Over Time', fontsize=14)
    ax2.set_xlabel('Record Date', fontsize=12)
    ax2.set_ylabel('Ratio', fontsize=12)
    ax2.legend()
    ax2.set_xticks(ax2.get_xticks()[::len(df)//num_ticks])
    ax2.tick_params(axis='x', rotation=45)

    plt.show(block=True)

def check_stationarity(df: pd.DataFrame):
    """
    Checks the stationarity of each column in the provided DataFrame using the Augmented Dickey-Fuller (ADF) test.
    
    Args:
        df (pandas.DataFrame): The DataFrame to check for stationarity.
    
    Returns:
        None
    """
    for i in range(len(df.columns)):
        result = adfuller(df[df.columns[i]])
        if result[1] < 0.05:
            print(f'{df.columns[i]} is stationary')
        else:
            print(f'{df.columns[i]} is not stationary')

def plot_seasonality(df: pd.DataFrame):
    """
    Performs a seasonal decomposition analysis on the 'Close' column of the provided DataFrame and plots the results.
    
    Args:
        df (pandas.DataFrame): A DataFrame containing the data to be analyzed, including a 'Close' column.
    
    Returns:
        None
    """
    result = seasonal_decompose(df['Close'],model='multiplicative')
    fig = plt.figure()
    fig = result.plot()
    fig.show()

def do_all(target: pd.DataFrame, train_exogenous: pd.DataFrame, test_exogenous: pd.DataFrame, full_dataset, full_target, trend):
    print('-'*100)
    print('Determining ARIMA Orders For Each Seasonality Steps Value')
    print('-'*100)
    season_orders = {}
    for i in range(2,13):
        model = auto_arima(target, X=train_exogenous, seasonal=True, m=i, suppress_warnings=True, trend=trend)
        print(i, model.order, model.seasonal_order)
        season_orders[i] = [model.order, model.seasonal_order]
    print('-'*100)
    print('Done')
    print('-'*100)

    print('-'*100)
    print('Training SARIMAX Models for each Order Found')
    print('-'*100)
    def train_plot_sarimax(train, exo_var, test, order: tuple, seasonal_order: tuple):
        model = SARIMAX(endog=train, exog=exo_var, order=order, seasonal_order=seasonal_order, trend=trend)
        fit = model.fit(disp=False)
        forecast = fit.get_forecast(steps=len(test), exog=test)
        forecast_mean = forecast.predicted_mean
        name = f'Seasonal {m}'
        return name, forecast_mean, fit.aic
    
    means = []
    for m in season_orders:
        print(m)
        name, cur, aic = train_plot_sarimax(train=target, exo_var=train_exogenous, test=test_exogenous,order=season_orders[m][0], seasonal_order=season_orders[m][1])
        means.append((name, cur, aic))
    sorted_means = sorted(means, key=lambda x: x[2])[:3]
    print('-'*100)
    print('Done')
    print('-'*100)

    cmap =  plt.get_cmap('tab10')
    forecast_index = pd.date_range(start=full_dataset.index[train_size - 1], periods=len(test_exogenous) + 1, freq='ME')[1:]
    plt.scatter(full_target.index, full_target, label='Actual Data', color='black', s=10)
    for i, pair in enumerate(sorted_means):
        name = pair[0] +' '+ str(pair[2])
        forecast_mean = pair[1]
        forecast_series = pd.Series(forecast_mean.values, index=forecast_index)
        plt.scatter(forecast_series.index, forecast_series, label=name, color=cmap(i%10), s=5)

    plt.title('USD to EUR Exchange Rate Forecast')
    plt.xlabel('Date')
    plt.ylabel('Close Exchange Rate')
    plt.legend()
    plt.show()
    print('-'*100)
    print('Done')
    print('-'*100)
    return sorted_means[:3]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('function', choices=['plot', 'model'])
    args = parser.parse_args()

    # set up dataframes
    economic_indicators = obtain_econ_data()
    modified_economic_indicators = modify_economic_data(economic_indicators)
    usd_eur = obtain_exchange_data()
    final = combine_both_dataframes(modified_economic_indicators, usd_eur)

    if args.function == 'plot':
        # plot data
        plot_us_indicators(final)
        plot_eur_indicators(final)
        plot_indicator_relationships(final)

    elif args.function == 'model':
        # split data into train and test
        train_size = int(len(final) * 0.9)
        train = final[:train_size] 
        test = final[train_size:] 

        # create possible exogenous variables
        exo1 = ['CCI_Difference',	
                'Inflation_Difference',
                'GDP_Difference',
                'Interest_Difference']
        exo2 = ['CCI_Ratio',
                'Inflation_Ratio',
                'GDP_Ratio',
                'Interest_Ratio']
        exo3 = ['CCI_Difference',	
                'Inflation_Difference',
                'GDP_Difference',
                'Interest_Difference',
                'CCI_Ratio',
                'Inflation_Ratio',
                'GDP_Ratio',
                'Interest_Ratio']
        exo4 = ['Euro CCI', 'US CCI', 'Euro Inflation Rate', 'US Inflation Rate',
                'Euro_GDP_Growth', 'US_GDP_Growth', 'US Interest Rate', 'Euro Interest Rate']
        exo5 = ['Euro CCI', 'US CCI', 'Euro Inflation Rate', 'US Inflation Rate',
                'Euro_GDP_Growth', 'US_GDP_Growth', 'US Interest Rate', 'Euro Interest Rate',
                'Inflation_Ratio','GDP_Ratio','Interest_Ratio']

        # find the best models
        top_3 = do_all(train['Close'], train[exo2], test[exo2], final, final['Close'], 'ct')
