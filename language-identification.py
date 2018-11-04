import nltk
import os
import random

def create_gramme(sentence,gramme_lenght):
    features = {}
    for i in range(len(sentence)-(gramme_lenght-1)):
        gramme = sentence[i:i+gramme_lenght]
        if gramme not in features:
            features[gramme] = 1
        else:
            features[gramme] += 1
    return features

def generate_features(gramme_lenght):
    feature_set = []
    language_class = 0
    for i in os.listdir("./identification_langue/identification_langue/corpus_entrainement"):
        feature_set.append([])
        with open("./identification_langue/identification_langue/corpus_entrainement/" + i, encoding="ISO-8859-1") as f:
            line = f.read()
            sentences = nltk.sent_tokenize(line)
            for sentence in sentences:
                features = create_gramme(sentence.lower(),gramme_lenght)
                feature_set[language_class].append((features,language_class))
        language_class += 1
    return feature_set

def generate_guessed_classes(classifier, features_set):
    guessed_classes = []
    for features in features_set:
        guessed_classes.append(classifier.classify(features))
    return guessed_classes

def generate_train_test_sets(sets):
    train_set, test_set = [],[]
    for i in range(len(sets)):
        random.shuffle(sets[i])
        size = len(sets[i])
        train_set = train_set + (sets[i][:int(size * 0.8)])
        test_set = test_set + (sets[i][int(size * 0.8):])
    random.shuffle(train_set)
    random.shuffle(test_set)
    return train_set, test_set

def calculate_stats(nb_of_classes,cm):
    true_positives = 0
    false_negatives = 0
    false_positives = 0

    for i in range(nb_of_classes):
        for j in range(nb_of_classes):
            if i == j:
                true_positives += cm[i,j]
            else:
                false_negatives += cm[i,j]
                false_positives += cm[i,j]
    return true_positives,false_negatives,false_positives

def print_stats(stats):
    true_positives = stats[0]
    false_negatives = stats[1]
    false_positives = stats[2]
    print("Accuracy = "+str(true_positives/float(true_positives+false_positives+false_negatives)))
    print("Precision = "+str(true_positives/float(true_positives+false_positives)))
    print("Recall = " + str(true_positives / float(true_positives + false_negatives)))


def main():
    features_set = generate_features(3)
    nb_of_classes = len(features_set)
    train_set, test_set = generate_train_test_sets(features_set)
    bayesClassifier = nltk.NaiveBayesClassifier.train(train_set)
    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    regClassifier = nltk.MaxentClassifier.train(train_set, algorithm, max_iter=3)
    element_list = list(zip(*test_set))
    test_features = list(element_list)[0]
    guessed_classes = generate_guessed_classes(bayesClassifier,test_features)
    test_class= list(element_list[1])
    cm = nltk.ConfusionMatrix(test_class, guessed_classes)
    print_stats(calculate_stats(nb_of_classes,cm))

    print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
    print(nltk.classify.accuracy(bayesClassifier, test_set))
    print(nltk.classify.accuracy(regClassifier, test_set))

if __name__ == "__main__":
   main()