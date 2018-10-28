# -*- coding: utf-8 -*-
import re
import nltk
import os
import random
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

PROVERBE_LANGUAGE = "french"
grammes = {}

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
    tokens = nltk.word_tokenize(line, PROVERBE_LANGUAGE)
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
    train_set, test_set = features_set[:int(size * 0.8)], features_set[int(size * 0.8):]
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
    normalization_methods = [no_normalization, stemming_normalization, lemmatize_normalization]
    train_and_test_classifier(classifier, lemmatize_normalization, get_all_features_without_stop_words, get_features_count)

if __name__ == "__main__":
   main()