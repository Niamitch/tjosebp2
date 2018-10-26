# -*- coding: utf-8 -*-
import re
import nltk
import os


PROVERBE_LANGUAGE = "french"
grammes = {}

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
        with open("./books/Book/neg_Bk/"+i) as f:
            get_sentiment_features(f.read())



if __name__ == "__main__":
   main()