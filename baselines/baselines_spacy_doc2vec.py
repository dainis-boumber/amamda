import numpy as np
import spacy as spacy
import pandas as pd
import logging
from pathlib import Path
import pickle

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import scorer

import baselines.prepare as prepare


def load_spacy_doc2vec():
    data_pickle = Path("av400_spacy_doc2vec.pickle")
    if not data_pickle.exists():

        nlp = spacy.load('en_vectors_web_lg')
        print("Spacy model loaded")
        (train_data, train_y), (val_data, val_y), (test_data, test_y) = prepare.load_dataframe()
        print("Text data loaded")

        logging.info("converting to spacy vector")
        train_data = train_data.applymap(lambda x: nlp(x).vector)
        val_data = val_data.applymap(lambda x: nlp(x).vector)
        test_data = test_data.applymap(lambda x: nlp(x).vector)
        logging.info("convert to spacy vector completed")

        pickle.dump([train_data, val_data, test_data, train_y, val_y, test_y], open(data_pickle, mode="wb"))
        logging.info("dumped all data structure in " + str(data_pickle))
    else:
        logging.info("loading spacy vector from PICKLE")
        [train_data, val_data, test_data, train_y, val_y, test_y] = pickle.load(open(data_pickle, mode="rb"))
        logging.info("load spacy vector completed")
    return (train_data, train_y), \
           (val_data, val_y), \
           (test_data, test_y)


def data_vector_sbs():
    (train_data, train_y), (val_data, val_y), (test_data, test_y) = load_spacy_doc2vec()

    train_vec = np.concatenate([np.stack(train_data['k_doc']), np.stack(train_data['u_doc'])], axis=1)
    val_vec = np.concatenate([np.stack(val_data['k_doc']), np.stack(val_data['u_doc'])], axis=1)
    test_vec = np.concatenate([np.stack(test_data['k_doc']), np.stack(test_data['u_doc'])], axis=1)
    return (train_vec, train_y), (val_vec, val_y), (test_vec, test_y)


def data_vector_diff():
    (train_data, train_y), (val_data, val_y), (test_data, test_y) = load_spacy_doc2vec()

    train_vec = np.stack(train_data['k_doc']) - np.stack(train_data['u_doc'])
    val_vec = np.stack(val_data['k_doc']) - np.stack(val_data['u_doc'])
    test_vec = np.stack(test_data['k_doc']) - np.stack(test_data['u_doc'])
    return (train_vec, train_y), (val_vec, val_y), (test_vec, test_y)


def main():
    logging.basicConfig(level=logging.INFO)

    clfs = [LinearSVC(), BernoulliNB(), SVC(kernel='rbf'), SVC(kernel='poly')]

    logging.info("SBS Vector TESTS =======================================")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_vector_sbs()
    for clf in clfs:
        logging.info("training " + type(clf).__name__)
        logging.info("param:" + str(clf.get_params()))

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = scorer.accuracy_score(y_test, pred)
        logging.info(acc)

    logging.info("DIFF Vector TESTS =======================================")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_vector_diff()
    for clf in clfs:
        logging.info("training " + type(clf).__name__)
        logging.info("param:" + str(clf.get_params()))

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        acc = scorer.accuracy_score(y_test, pred)
        logging.info(acc)


if __name__ == '__main__':
    main()
