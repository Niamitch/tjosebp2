# -*- coding: utf-8 -*-
import re
import nltk
import os
import random
import numpy as np

PROVERBE_LANGUAGE = "french"
grammes = {}
featuresSet = []
def get_sentiment_features(line):
    features = {}
    tokens = nltk.word_tokenize(line, PROVERBE_LANGUAGE)
    for token in tokens:
        if token not in features:
            features[token] = 1
        else:
            features[token] +=1

    return features

def main():
    for i in os.listdir("./books/Book/neg_Bk"):
        with open("./books/Book/neg_Bk/"+i, encoding = "ISO-8859-1") as f:
            features = get_sentiment_features(f.read())
            featuresSet.append((features,0))
    for i in os.listdir("./books/Book/pos_Bk"):
        with open("./books/Book/pos_Bk/"+i, encoding = "ISO-8859-1") as f:
            features = get_sentiment_features(f.read())
            featuresSet.append((features, 1))
    random.shuffle(featuresSet)
    size = len(featuresSet)
    train_set, test_set = featuresSet[int(size*0.8):], featuresSet[:int(size*0.2)]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))



if __name__ == "__main__":
   main()