import nltk
import os
import random

# bigramme beaucoup plus stable que trigramme pour une raison obscure
def create_gramme(sentence):
    features = {}
    for i in range(len(sentence)-2):
        gramme = sentence[i:i+3]
        if gramme not in features:
            features[gramme] = 1
        else:
            features[gramme] += 1
    return features

def generate_features():
    feature_set = []
    language_class = 0
    for i in os.listdir("./identification_langue/identification_langue/corpus_entrainement"):
        with open("./identification_langue/identification_langue/corpus_entrainement/" + i, encoding="ISO-8859-1") as f:
            line = f.read()
            sentences = nltk.sent_tokenize(line)
            for sentence in sentences:
                features = create_gramme(sentence.lower())
                feature_set.append((features,language_class))
        language_class += 1
    return feature_set



def main():
    features_set = generate_features()
    random.shuffle(features_set)
    size = len(features_set)
    train_set, test_set = features_set[:int(size * 0.8)], features_set[int(size * 0.8):]
    bayesClassifier = nltk.NaiveBayesClassifier.train(train_set)
    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    regClassifier = nltk.MaxentClassifier.train(train_set, algorithm, max_iter=3)
    print(nltk.classify.accuracy(bayesClassifier, test_set))
    print(nltk.classify.accuracy(regClassifier, test_set))

if __name__ == "__main__":
   main()