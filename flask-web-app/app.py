import flask
import os
import pickle
import pandas as pd
from flask import app
from news_scraper import news_headlines

app = flask.Flask(__name__, template_folder='templates')
path_to_vectorizer = 'models/vectorizer.pkl'
path_to_classifier = 'models/movement-classifier.pkl'

with open(path_to_vectorizer, 'rb') as f:
    vectorizer = pickle.load(f)

with open(path_to_classifier, 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET','POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))

    if flask.request.method == 'POST':
        ticker = flask.request.form['ticker']
        text = news_headlines(ticker)

        X = vectorizer.transform([text])

        prediction = model.predict(X)

        return flask.render_template('index.html',
                                     input_text= ticker,
                                     result= prediction)

if __name__ == '__main__':
    app.run(debug=True)










