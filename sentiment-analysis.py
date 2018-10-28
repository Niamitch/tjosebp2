# -*- coding: utf-8 -*-
import re
import nltk
import os
import random
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.metrics.scores import (precision, recall)
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

tfidf = None
tfidf_matrix = None
neg_corpus = []
pos_corpus = []

#Normalization function
def no_normalization(token):
    return token

def stemming_normalization(token):
    ps = PorterStemmer()
    return ps.stem(token)

def lemmatize_normalization(token):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(token)

# Attribute selection functions
def get_all_features(normalization_function, attribute_value_selection_function, line):
    tokens = nltk.word_tokenize(line, 'english')
    filtered_tokens = []
    for token in tokens:
        pattern = re.compile("\w*")
        if(pattern.match(token)):
            filtered_tokens.append(token)
    return attribute_value_selection_function(filtered_tokens, normalization_function)

def get_all_features_with_frequency_upper_than_one(normalization_function, attribute_value_selection_function, line):
    feature_count_by_feature = get_all_features(normalization_function, attribute_value_selection_function, line)
    filtered_feature_count_by_feature = {}
    for feature in feature_count_by_feature:
        if feature_count_by_feature[feature] > 1:
            filtered_feature_count_by_feature[feature] = feature_count_by_feature[feature]
    return filtered_feature_count_by_feature

def get_all_features_without_stop_words(normalization_function, attribute_value_selection_function, line):
    feature_count_by_feature = get_all_features(normalization_function, attribute_value_selection_function, line)
    stop_words = set(stopwords.words('english'))
    filtered_feature_count_by_feature = {}
    for feature in feature_count_by_feature:
        if feature not in stop_words:
            filtered_feature_count_by_feature[feature] = feature_count_by_feature[feature]
    return filtered_feature_count_by_feature

def get_all_features_with_open_class(normalization_function, attribute_value_selection_function, line):
    feature_count_by_feature = get_all_features(normalization_function, attribute_value_selection_function, line)
    tagged_words = pos_tag(line)
    filtered_feature_count_by_feature = {}
    open_class_words = [word for word, tag in tagged_words if tag in
                        ('RB', 'RBR', 'RBS' ,'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ' ,'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS',)]
    for feature in feature_count_by_feature:
        if feature in open_class_words:
            filtered_feature_count_by_feature[feature] = feature_count_by_feature[feature]
    return filtered_feature_count_by_feature

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

def get_features_tf_idf(tokens, normalization_function):
    features_tf_idf = {}
    for token in tokens:
        token = normalization_function(token)
        if token not in features_tf_idf and token in tfidf.vocabulary_:
            features_tf_idf[token] = tfidf_matrix[0, tfidf.vocabulary_[token]]
    return features_tf_idf

def get_features_presence(tokens, normalization_function):
    features_count_by_feature = get_features_count(tokens, normalization_function)
    feature_presence = {}
    for feature in features_count_by_feature:
        feature_presence[feature] = 1
    return feature_presence

# Execution code
def get_features(corpus, output_class, normalization_function, attribute_selection_function, attribute_value_selection_function):
    features_set = []
    for line in corpus:
        features = attribute_selection_function(normalization_function, attribute_value_selection_function, line)
        features_set.append((features, output_class))
    return features_set

def get_train_test_sets(features_set):
    random.shuffle(features_set)
    size = len(features_set)
    train_set, test_set = features_set[:int(size * 0.8)], features_set[int(size * 0.8):]
    return train_set, test_set


def load_corpus():
    for i in os.listdir("./books/Book/neg_Bk"):
        with open("./books/Book/neg_Bk/"+i, encoding = "ISO-8859-1") as f:
            line = f.read()
            neg_corpus.append(line)
    for i in os.listdir("./books/Book/pos_Bk"):
        with open("./books/Book/pos_Bk/"+i, encoding = "ISO-8859-1") as f:
            line = f.read()
            pos_corpus.append(line)

    global tfidf
    tfidf = TfidfVectorizer()
    global tfidf_matrix
    tfidf_matrix = tfidf.fit_transform(neg_corpus + pos_corpus)


def train_and_test_classifier(classifier, normalization_function, attribute_selection_function, attribute_value_selection_function):
    features_set = []
    features_set = features_set + get_features(neg_corpus, 0, normalization_function, attribute_selection_function, attribute_value_selection_function)
    features_set = features_set + get_features(pos_corpus, 1, normalization_function, attribute_selection_function, attribute_value_selection_function)
    train_set, test_set = get_train_test_sets(features_set)
    classifier = classifier.train(train_set)
    print('Accuracy: ' + str(nltk.classify.accuracy(classifier, test_set)))

def main():
    load_corpus()
    classifier = nltk.NaiveBayesClassifier
    normalization_methods = [no_normalization, stemming_normalization, lemmatize_normalization]
    feature_selection_methods = [get_all_features, get_all_features_with_frequency_upper_than_one, get_all_features_without_stop_words, get_all_features_with_open_class]
    train_and_test_classifier(classifier, lemmatize_normalization, get_all_features, get_features_tf_idf)

if __name__ == "__main__":
   main()