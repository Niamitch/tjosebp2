import nltk
import os
import random
from sklearn.metrics import precision_score, recall_score

def create_gramme(sentence,gramme_lenght):
    features = {}
    for i in range(len(sentence)-(gramme_lenght-1)):
        gramme = sentence[i:i+gramme_lenght]
        if gramme not in features:
            features[gramme] = 1
        else:
            features[gramme] += 1
    return features

def get_class_name(file):
    return file.split('-')[0][:2]

def generate_features(gramme_lenght):
    feature_set = []
    language_class = 0
    for i in os.listdir("./identification_langue/identification_langue/corpus_entrainement"):
        feature_set.append([])
        with open("./identification_langue/identification_langue/corpus_entrainement/" + i, encoding="ISO-8859-1") as f:
            line = f.read()
            class_name = get_class_name(i)
            sentences = nltk.sent_tokenize(line)
            document = []
            for i in range(len(sentences)):
                document.append(create_gramme(sentences[i].lower(),gramme_lenght))
                if (i+1) % 3 == 0:
                    features = { k: document[0].get(k, 0) + document[1].get(k, 0) + document[2].get(k, 0) for k in set(document[0]) | set(document[1]) | set(document[2])}
                    feature_set[language_class].append((features, class_name))
                    document = []
            if len(document) == 2:
                features = {k: document[0].get(k, 0) + document[1].get(k, 0) for k in set(document[0]) | set(document[1])}
                feature_set[language_class].append((features, class_name))
            if len(document) == 1:
                feature_set[language_class].append((document[0], class_name))
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

def print_stats(classifier,test_set,real_test_classes,predicted_test_classes):
    cm = nltk.ConfusionMatrix(real_test_classes, predicted_test_classes)
    print('Accuracy: ' + str(nltk.classify.accuracy(classifier, test_set)))
    print('Precision: ' + str(precision_score(real_test_classes, predicted_test_classes, average='micro')))
    print('Recall: ' + str(recall_score(real_test_classes, predicted_test_classes, average='micro')))
    print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))


def main():
    features_set = generate_features(3)
    train_set, test_set = generate_train_test_sets(features_set)
    bayesClassifier = nltk.NaiveBayesClassifier.train(train_set)
    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    regClassifier = nltk.MaxentClassifier.train(train_set, algorithm, max_iter=3)
    element_list = list(zip(*test_set))
    test_features = list(element_list)[0]

    bayesian_guessed_classes = generate_guessed_classes(bayesClassifier,test_features)
    reg_guessed_classes = generate_guessed_classes(regClassifier,test_features)
    test_class= list(element_list[1])

    print("Bayes stats:")
    print_stats(bayesClassifier,test_set,test_class,bayesian_guessed_classes)
    print("Regs stats:")
    print_stats(regClassifier, test_set, test_class, reg_guessed_classes)



if __name__ == "__main__":
   main()