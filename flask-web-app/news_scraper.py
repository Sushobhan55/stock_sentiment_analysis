from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import time
from text_pipeline import text_pipeline

todays_date = time.strftime("%b-%d-%y")

def news_headlines(ticker):
    news_table = {}
    url = 'https://finviz.com/quote.ashx?t=' + ticker
    req = Request(url=url,headers={'user-agent': 'my-app'})
    response = urlopen(req)

    html = BeautifulSoup(response, features='html.parser')
    news_table = html.find(id='news-table')

    parsed_news = []
    for row in news_table.findAll('tr'):
        text = row.a.text
        date_data = row.td.text.split(' ')
        link = row.a["href"]

        if len(date_data) == 1:
            time = date_data[0]

        else:
            date = date_data[0]
            time = date_data[1]

        if date == todays_date:
            parsed_news.append([date, time, text, link])

    df = pd.DataFrame(parsed_news)
    df.columns = ['date', 'time', 'title', 'link']

    text = ""
    for title in df.title:
        text = text + title

    text_clean = text_pipeline(text)
    print(text_clean)

