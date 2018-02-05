import logging

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import scorer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import baselines.prepare as prep

vector_comparison_methods = [prep.data_vector_sbs,
                             prep.data_vector_diff]

vectorization_methods = {"char binary": CountVectorizer(binary=True, analyzer='char'),
                         "char 1gram count": CountVectorizer(binary=False, analyzer='char'),
                         "char 1-5gram count": CountVectorizer(binary=False, analyzer='char', ngram_range=(1, 5)),
                         "char 1gram tfidf": TfidfVectorizer(analyzer='char'),
                         "char 1-5gram tfidf": TfidfVectorizer(analyzer='char', ngram_range=(1, 5)),

                         "word binary": CountVectorizer(binary=True, analyzer='word'),
                         "word 1gram count": CountVectorizer(binary=False, analyzer='word'),
                         "word 1-3gram count": CountVectorizer(binary=False, analyzer='word', ngram_range=(1, 3)),
                         "word 1gram tfidf": TfidfVectorizer(analyzer='word', ngram_range=(1, 1)),
                         "word 1-3gram tfidf": TfidfVectorizer(analyzer='word', ngram_range=(1, 3))
                         }


def all_sbs():
    for method_name, vec_method in vectorization_methods.items():
        logging.info("Vectorization Method: " + method_name)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prep.data_vector_sbs(vec_method)
        clfs = [LinearSVC(), BernoulliNB()]

        for clf in clfs:
            logging.info("training " + type(clf).__name__)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            acc = scorer.accuracy_score(y_test, pred)
            print(acc)


def all_diff():
    for method_name, vec_method in vectorization_methods.items():
        logging.info("Vectorization Method: " + method_name)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prep.data_vector_diff(vec_method)
        clfs = [LinearSVC(), BernoulliNB()]

        for clf in clfs:
            logging.info("training " + type(clf).__name__)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            acc = scorer.accuracy_score(y_test, pred)
            logging.info("ACC: " + str(acc))


def try_one():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) =\
        prep.data_vector_diff(CountVectorizer(binary=False, analyzer='word'))
    model = LinearSVC()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = scorer.accuracy_score(y_test, pred)
    logging.info("ACC: " + str(acc))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    try_one()
