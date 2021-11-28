import flask
import pickle
from text_pipeline import text_pipeline
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import time
from chart import price_chart

todays_date = time.strftime("%b-%d-%y")

app = flask.Flask(__name__, template_folder='templates')
path_to_vectorizer = 'models/vectorizer.pkl'
path_to_classifier = 'models/movement-classifier.pkl'

with open(path_to_vectorizer, 'rb') as f:
    vectorizer = pickle.load(f)

with open(path_to_classifier, 'rb') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('index.html'))

    if flask.request.method == 'POST':
        ticker = flask.request.form['ticker']
        predictions = []

        news_table = {}
        url = 'https://finviz.com/quote.ashx?t=' + ticker
        req = Request(url=url, headers={'user-agent': 'my-app'})
        response = urlopen(req)

        html = BeautifulSoup(response, features='html.parser')
        news_table = html.find(id='news-table')

        parsed_news = []
        text_clean = []
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

        predictions = []
        clean_text = []
        for i in range(len(df.title)):
            text = text_pipeline(df.title[i])
            clean_text.append(text)
        X = vectorizer.transform(clean_text)
        prediction = model.predict(X)
        predictions.append(prediction)


        df["prediction"] = predictions[0].tolist()

        return flask.render_template('index.html',
                                     input_text=ticker.upper(),
                                     plot = price_chart(ticker),
                                     table=df.values.tolist(),
                                     headings=df.columns.values)


if __name__ == '__main__':
    app.run(debug=True)
