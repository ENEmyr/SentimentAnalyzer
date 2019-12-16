# -*- coding: utf-8 -*-
import numpy as np
import codecs
from pythainlp.tokenize import word_tokenize
from collections import defaultdict

class NaiveBayesClassifier():

    def __init__(self):
        self.prior = defaultdict(int)
        self.log_prior = {}
        self.assem_docs = defaultdict(list)
        self.log_likelihoods = defaultdict(defaultdict)
        self.vocabs = []

    def _build_vocab_set(self, docs):
        vocabs = set()
        for doc in docs:
            for word in self.tokenize(doc):
                vocabs.add(word)
        return vocabs

    def _n_word_each_class(self):
        counts = {}
        for c in list(self.assem_docs.keys()):
            docs = self.assem_docs[c]
            counts[c] = defaultdict(int)
            for doc in docs:
                words = self.tokenize(doc)
                for word in words:
                    counts[c][word] += 1

        return counts

    def train(self, training_set, alpha=1):
        '''
        Train the computer with given training_set
        :param training_set:
            list of zip(training_samples, training_labels)
        :param alpha:
            additive smoothing or Laplace smoothing
        '''
        count_docs = len(training_set)
        samples, labels = [], []
        for sample, label in training_set:
            samples.append(sample)
            labels.append(label)

        self.vocabs = self._build_vocab_set(samples)

        for sample, label in training_set:
            self.assem_docs[label].append(sample)

        all_classes = set(labels)

        self.word_count = self._n_word_each_class()

        for c in all_classes:
            count_docs_in_class = float(sum(np.array(labels) == c))
            self.log_prior[c] = np.log(count_docs_in_class / count_docs)

            total_word = 0 # total_word in c class
            for word in self.vocabs:
                total_word += self.word_count[c][word]

            for word in self.vocabs:
                each_word_count = self.word_count[c][word]
                self.log_likelihoods[c][word] = np.log((each_word_count + alpha) / (total_word + alpha * len(self.vocabs)))

    def predict(self, text):
        '''
        Predict given sentence
        :param text:
            sentence or document that need to be predict
        '''
        predict_result = { 0: 0, 1: 0 } # { 'neg': 0, 'pos': 0}
        for c in self.assem_docs.keys():
            predict_result[c] = self.log_prior[c]
            words = self.tokenize(text)
            for word in words:
               if word in self.vocabs:
                   predict_result[c] += self.log_likelihoods[c][word]
        return predict_result

    def tokenize(self, text):
        return word_tokenize(text)

if __name__ == "__main__":
    NB_classifier = NaiveBayesClassifier()
    with codecs.open('../SentimentLexicon/pos.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    pos_list = [e.strip() for e in lines]
    del lines
    f.close()
    with codecs.open('../SentimentLexicon/neg.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()
    neg_list = [e.strip() for e in lines]
    del lines
    f.close()
    pos_labels = [1]*len(pos_list)
    neg_labels = [0]*len(neg_list)
    training_set = list(zip(pos_list, pos_labels)) + list(zip(neg_list, neg_labels))

    NB_classifier.train(training_set)
    while True:
        text = input()
        predict_result = NB_classifier.predict(text)
        if predict_result[0] >= predict_result[1]:
            print('negative')
        else:
            print('positive')