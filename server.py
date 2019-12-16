# -*- coding: utf-8 -*-
import codecs
from NaiveBayesSentiment.NaiveBayes import NaiveBayesClassifier
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
NBClassifier = NaiveBayesClassifier()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asdzxczx5%Axzc000'
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/sentiment')
@cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def Home():
    return '<h1>Welcome to Sentiment Analyzer</h1><br><h2>Send the request to https://127.0.0.1:3000/sentiment/"sentence" for test the anlyzer</h2>'

@app.route('/sentiment/<string:sentence>')
@cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def Sentiment(sentence):
    return_result = ''
    predict_result = NBClassifier.predict(sentence)
    if predict_result[0] >= predict_result[1]:
        return_result = 'negative'
    else:
        return_result = 'positive'
    return '%s' % return_result

if __name__ == '__main__':
    with codecs.open('SentimentLexicon/pos.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    pos_list = [e.strip() for e in lines]
    del lines
    f.close()
    with codecs.open('SentimentLexicon/neg.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    neg_list = [e.strip() for e in lines]
    del lines
    f.close()
    pos_labels = [1]*len(pos_list)
    neg_labels = [0]*len(neg_list)
    training_set = list(zip(pos_list, pos_labels)) + list(zip(neg_list, neg_labels))

    NBClassifier.train(training_set)
    print("Sentiment Analyzer loaded.")

    app.run(
        debug=True,
        host='127.0.0.1',
        port='3000'
    )
