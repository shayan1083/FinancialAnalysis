import numpy as np
import pandas as pd
import yfinance as yf
import mysql.connector as sql
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import praw
from time import strftime, localtime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def open_sql_connection():
    load_dotenv()
    return create_engine(f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASS')}@localhost/stock_data")

def create_reddit_connection():
    load_dotenv()
    return praw.Reddit(
                    client_id=os.getenv('REDDIT_CLIENT_ID'),
                    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                    user_agent=os.getenv('REDDIT_USER_AGENT')
                    )
    
def gather_financial_data(ticker: str):
    msft = yf.Ticker(ticker)
    hist = msft.history(period="5y")
    fin_data = pd.DataFrame(hist)
    fin_data.drop(columns=['Dividends','Stock Splits'], inplace=True)
    fin_data.index = pd.to_datetime(fin_data.index).date
    df = fin_data.reset_index()
    df.columns = ['record_date','open_price','high_price','low_price','close_price','volume']
    engine = open_sql_connection()
    df.to_sql(name='financial_data', con=engine, if_exists='replace', index=False)
  
def gather_headline_data(ticker: str):
    reddit = create_reddit_connection()
    subreddit = reddit.subreddit('stocks+investing')
    company = yf.Ticker(ticker)
    company_name = company.info['longName']
    search_params = {
                    'query':f'title:"{ticker}" OR title:"{company_name}"',
                    'sort':'relevance',
                    'syntax':'lucene',
                    'time_filter':'all'
                    }
    generator_params = { 
                    'limit':1000
                    }
    posts = []
    for submission in subreddit.search(**search_params, **generator_params):
        date = strftime('%Y-%m-%d', localtime(submission.created_utc))
        posts.append({
            'date': date,
            'title': submission.title,
            'score': submission.score,
            'url': submission.url,  
        })
    tex_data = pd.DataFrame(posts)
    tex_data['date'] = pd.to_datetime(tex_data['date']).dt.date
    engine = open_sql_connection()
    tex_data.to_sql(name='text_information', con=engine, if_exists='replace', index=False)

def sentiment_per_date():
    engine = open_sql_connection()
    query = '''
        select date, title
        from text_information
    '''
    df = pd.read_sql(query,engine)
    analyzer = SentimentIntensityAnalyzer()
    def get_score(post_title):
        sentiment = analyzer.polarity_scores(post_title)
        return sentiment['compound']
    df['sentiment_score'] = df['title'].apply(get_score)
    df_sentiment = df.groupby('date')['sentiment_score'].mean().reset_index()
    df_grouped = df.groupby('date')['title'].apply(list).reset_index()
    df_final = pd.merge(df_grouped, df_sentiment, on='date')
    df_final[['date', 'sentiment_score']].to_sql('date_score', con=engine, if_exists='replace', index=False)

def prepare_stock_time_series():
    engine = open_sql_connection()
    query = '''
    SELECT fd.record_date, fd.open_price, fd.high_price, fd.low_price, fd.close_price, fd.volume, COALESCE(ds.sentiment_score, 0) AS score
    FROM financial_data AS fd
    LEFT JOIN date_score AS ds
    ON fd.record_date = ds.date;
    '''
    df = pd.read_sql(query, engine)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    df.to_sql(name='time_series_data', con=engine, if_exists='replace', index=False)

def read_CCI_data():
    euro_cci_path = 'ConsumerConfidence/EuroCCI.xlsx'
    euro_cci = pd.read_excel(euro_cci_path, skiprows=1)
    e_cci = euro_cci[['Category', 'Economic Sentiment Indicator (ESI)']]
    e_cci.rename(columns={'Category': 'DATE', 'Economic Sentiment Indicator (ESI)': 'Euro CCI'}, inplace=True)
    e_cci['DATE'] = pd.to_datetime(e_cci['DATE'])
    e_cci['DATE'] = e_cci['DATE'].dt.to_period('M')

    us_cci_path = 'ConsumerConfidence/USCCI.xlsx'
    us_cci = pd.read_excel(us_cci_path, skiprows=10)
    u_cci = us_cci[['observation_date', 'UMCSENT']]
    u_cci.rename(columns={'observation_date': 'DATE', 'UMCSENT': 'US CCI'}, inplace=True)
    u_cci['DATE'] = pd.to_datetime(u_cci['DATE'])
    u_cci['DATE'] = u_cci['DATE'].dt.to_period('M')
    return e_cci, u_cci

def read_inflation_data():
    euro_inflation_path = 'InflationRate/EuroInflationRate.xlsx'
    euro_inflation = pd.read_excel(euro_inflation_path, skiprows=1)
    e_inflation = euro_inflation[['DATE', 'HICP - Overall index (ICP.M.U2.N.000000.4.ANR)']]
    e_inflation.rename(columns={'HICP - Overall index (ICP.M.U2.N.000000.4.ANR)': 'Euro Inflation Rate'}, inplace=True)
    e_inflation['DATE'] = pd.to_datetime(e_inflation['DATE'])
    e_inflation['DATE'] = e_inflation['DATE'].dt.to_period('M')

    us_inflation_path = 'InflationRate/USInflationRate.xlsx'
    us_inflation = pd.read_excel(us_inflation_path, skiprows=11)
    u_inflation = pd.melt(us_inflation, id_vars=['Year'], var_name='Month', value_name='US Inflation Rate')
    u_inflation = u_inflation[~u_inflation['Month'].isin(['HALF1', 'HALF2'])]
    u_inflation['DATE'] = pd.to_datetime(u_inflation['Year'].astype(str) + u_inflation['Month'], format='%Y%b')
    u_inflation = u_inflation.drop(columns=['Year', 'Month'])
    u_inflation = u_inflation.sort_values(by='DATE')
    u_inflation['DATE'] = u_inflation['DATE'].dt.to_period('M')
    date = u_inflation.pop('DATE')
    u_inflation.insert(0, 'DATE', date)

    return e_inflation, u_inflation

def read_interest_rate_data():
    euro_interest_rate_path = 'InterestRates/EuroInterestRate.xlsx'
    euro_interest_rate = pd.read_excel(euro_interest_rate_path, skiprows=14)
    e_interest = euro_interest_rate[['DATE', 'Main refinancing operations - fixed rate tenders (fixed rate) (date of changes) - Level (FM.B.U2.EUR.4F.KR.MRR_FR.LEV)']]
    e_interest.rename(columns={'Main refinancing operations - fixed rate tenders (fixed rate) (date of changes) - Level (FM.B.U2.EUR.4F.KR.MRR_FR.LEV)': 'Euro Interest Rate'}, inplace=True)
    e_interest['DATE'] = pd.to_datetime(e_interest['DATE'])
    e_interest['DATE'] = e_interest['DATE'].dt.to_period('M')

    us_interest_rate_path = 'InterestRates/USInterestRate.xlsx'
    us_interest_rate = pd.read_excel(us_interest_rate_path, skiprows=10)
    u_interest = us_interest_rate[['observation_date', 'FEDFUNDS']]
    u_interest.rename(columns={'observation_date': 'DATE', 'FEDFUNDS': 'US Interest Rate'}, inplace=True)
    u_interest['DATE'] = pd.to_datetime(u_interest['DATE'])
    u_interest['DATE'] = u_interest['DATE'].dt.to_period('M')
    
    return e_interest, u_interest

def combine_economic_data():
    e_cci, u_cci = read_CCI_data()
    e_inflation, u_inflation = read_inflation_data()
    e_interest, u_interest = read_interest_rate_data()
    combined_df = e_cci.merge(u_cci, on='DATE')\
                      .merge(e_inflation, on='DATE')\
                      .merge(u_inflation, on='DATE')\
                      .merge(e_interest, on='DATE', how='outer')\
                      .merge(u_interest, on='DATE')
    combined_df = combined_df.rename(columns={'DATE': 'record_date'})
    combined_df['record_date'] = combined_df['record_date'].dt.to_timestamp().dt.date
    engine = open_sql_connection()
    combined_df.to_sql(name='economic_indicators', con=engine, if_exists='replace', index=False)

if __name__ == '__main__':
    combine_economic_data()
    # gather_financial_data('MSFT')
    # gather_headline_data('MSFT')
    # sentiment_per_date()
    # prepare_stock_time_series()