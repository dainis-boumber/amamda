from pathlib import Path
import pickle
import logging

import numpy as np
from scipy import sparse
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import data.MLP400AV.mlpapi as mlpapi


def load(schema='A2', path_to_ds='../data/MLP400AV/'):
    loader = mlpapi.MLPVLoader(schema, fileformat='pandas', directory=path_to_ds)
    train, val, test = loader.get_mlpv()
    return train, val, test


def load_dataframe():
    data_pickle = Path("av400tuple.pickle")
    if not data_pickle.exists():
        logging.info("loading data structure from RAW")
        train_data, val_data, test_data = load()

        train_y = train_data['label'].tolist()
        val_y = val_data['label'].tolist()
        test_y = test_data['label'].tolist()

        train_data.drop(['k_author', 'u_author', 'label'], axis=1, inplace=True)
        val_data.drop(['k_author', 'u_author', 'label'], axis=1, inplace=True)
        test_data.drop(['k_author', 'u_author', 'label'], axis=1, inplace=True)

        logging.info("load data structure completed")

        pickle.dump([train_data, val_data, test_data, train_y, val_y, test_y], open(data_pickle, mode="wb"))
        logging.info("dumped all data structure in " + str(data_pickle))
    else:
        logging.info("loading data structure from PICKLE")
        [train_data, val_data, test_data, train_y, val_y, test_y] = pickle.load(open(data_pickle, mode="rb"))
        logging.info("load data structure completed")

    return (train_data, train_y), (val_data, val_y), (test_data, test_y)


def transform_tuple(X_train, X_val, X_test, vectorizer:CountVectorizer):
    vectorizer.fit(X_train['k_doc'].append(X_train['u_doc']))
    train = X_train.apply(vectorizer.transform)
    val = X_val.apply(vectorizer.transform)
    test = X_test.apply(vectorizer.transform)
    return train, val, test


def data_vector_sbs(vectorizer):
    (train_data, train_y), (val_data, val_y), (test_data, test_y) = load_dataframe()
    train_vec, val_vec, test_vec = transform_tuple(train_data, val_data, test_data, vectorizer)
    train_vec = sparse.hstack((train_vec['k_doc'], train_vec['u_doc'])).tocsr()
    val_vec = sparse.hstack((val_vec['k_doc'], val_vec['u_doc'])).tocsr()
    test_vec = sparse.hstack((test_vec['k_doc'], test_vec['u_doc'])).tocsr()
    return (train_vec, train_y), (val_vec, val_y), (test_vec, test_y)


def data_vector_diff(vectorizer):
    (train_data, train_y), (val_data, val_y), (test_data, test_y) = load_dataframe()
    train_vec, val_vec, test_vec = transform_tuple(train_data, val_data, test_data, vectorizer)
    train_vec = (train_vec['k_doc'] - train_vec['u_doc']).tocsr()
    val_vec = (val_vec['k_doc'] - val_vec['u_doc']).tocsr()
    test_vec = (test_vec['k_doc'] - test_vec['u_doc']).tocsr()
    return (train_vec, train_y), (val_vec, val_y), (test_vec, test_y)


def main():
    pass


if __name__ == '__main__':
    main()
