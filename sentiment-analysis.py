# -*- coding: utf-8 -*-
import re
import nltk
import os
import random
import numpy as np
nltk.download('punkt')

PROVERBE_LANGUAGE = "french"
grammes = {}

#Normalization function
def no_normalization(token):
    return token

# Attribute selection functions
def get_all_features(normalization_function, attribute_value_selection_function, line):
    features = {}
    tokens = nltk.word_tokenize(line, PROVERBE_LANGUAGE)
    filtered_tokens = []
    for token in tokens:
        pattern = re.compile("\w*")
        if(pattern.match(token)):
            filtered_tokens.append(token)
    return attribute_value_selection_function(filtered_tokens, normalization_function)

#Attribute value selection functions
def get_features_count(tokens, normalization_function):
    feature_counts = {}
    for token in tokens:
        token = normalization_function(token)
        if token not in feature_counts:
            feature_counts[token] = 1
        else:
            feature_counts[token] += 1
    return feature_counts

def get_features(folder, output_class, normalization_function, attribute_selection_function, attribute_value_selection_function):
    features_set = []
    for i in os.listdir(folder):
        with open(folder+"/"+i, encoding = "ISO-8859-1") as f:
            features = attribute_selection_function(normalization_function, attribute_value_selection_function, f.read())
            features_set.append((features, output_class))
    return features_set

def get_train_test_sets(features_set):
    random.shuffle(features_set)
    size = len(features_set)
    train_set, test_set = features_set[int(size * 0.8):], features_set[:int(size * 0.2)]
    return train_set, test_set

def train_and_test_classifier(classifier, normalization_function, attribute_selection_function, attribute_value_selection_function):
    features_set = []
    features_set = features_set + get_features("./books/Book/neg_Bk", 0, normalization_function, attribute_selection_function, attribute_value_selection_function)
    features_set = features_set + get_features("./books/Book/pos_Bk", 1, normalization_function, attribute_selection_function, attribute_value_selection_function)
    train_set, test_set = get_train_test_sets(features_set)
    classifier = classifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))

def main():
    classifier = nltk.NaiveBayesClassifier
    train_and_test_classifier(classifier, no_normalization, get_all_features, get_features_count)

if __name__ == "__main__":
   main()