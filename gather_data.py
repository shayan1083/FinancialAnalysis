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
    subreddit = reddit.subreddit('all')
    search_params = {
                    'query':f'title:"{ticker}"',
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

def prepare_time_series():
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

if __name__ == '__main__':
    gather_financial_data('MSFT')
    gather_headline_data('MSFT')
    sentiment_per_date()
    prepare_time_series()