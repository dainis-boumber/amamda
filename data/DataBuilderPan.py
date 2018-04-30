import logging
from pathlib import Path
from multiprocessing import Pool

import pickle
import numpy as np
import pandas as pd
import textacy

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from data.base import DataObject, PANData, DataBuilder
from utils.preprocessing.clean import clean_text_minor
from utils.preprocessing.clean import clean_text_major


class DataBuilderPan(DataBuilder):

    def __init__(self, year, train_split, test_split, embed_dim, vocab_size, target_doc_len, target_sent_len,
                 sent_split=False, word_split=False):
        super(DataBuilderPan, self).__init__(embed_dim=embed_dim, vocab_size=vocab_size,
                                               target_doc_len=target_doc_len, target_sent_len=target_sent_len)
        logging.info("YEAR: %s", year)
        self.year = year
        logging.info("TRAIN SPLIT: %s", train_split)
        self.train_split = train_split
        logging.info("TEST SPLIT: %s", test_split)
        self.test_split = test_split
        logging.info("setting: %s is %s", "sent_split", sent_split)
        self.sent_split = sent_split
        logging.info("setting: %s is %s", "word_split", word_split)
        self.word_split = word_split
        self.problem_name = "PAN" + self.year
        self.dataset_dir = self.data_path + self.problem_name
        self.num_classes = 2  # true or false
        self.domain_list = None
        if self.problem_name == 'PAN15':
            path_to_pickle = "PAN15tuple.pickle"
        elif self.problem_name == 'PAN14':
            path_to_pickle = 'PAN14tuple.pickle'
        elif self.problem_name == 'PAN13':
            path_to_pickle = 'PAN13tuple.pickle'
        else:
            raise NotImplementedError
        self.load_and_proc_data(path_to_pickle)

    def load_and_proc_data(self, path_to_pickle):
        (train_data, train_y), (test_data, test_y) = self.load_dataframe(path_to_pickle)

        all_data = pd.concat([train_data, test_data])
        uniq_doc = pd.unique(all_data.values.ravel('K'))
        logging.info("Total NO. of Unique Document: " + str(len(uniq_doc)))

        # Word_split result in word level embedding
        if self.word_split:
            doc_vec_dict, tokenizer = self.make_word_doc_vector(uniq_doc)

            train_vector = train_data.applymap(lambda x: doc_vec_dict[x])
            test_vector = test_data.applymap(lambda x: doc_vec_dict[x])

            self.train_data = self.make_data_obj(train_data, train_vector, train_y)
            self.test_data = self.make_data_obj(test_data, test_vector, test_y)
            self.vocab = tokenizer.word_index
            self.embed_matrix =  self.build_embedding_matrix()

        # This would be char level 1-hot encoding
        else:
            doc_vec_dict, tokenizer = self.make_char_doc_vector(uniq_doc)

            train_vector = train_data.applymap(lambda x: doc_vec_dict[x])
            test_vector = test_data.applymap(lambda x: doc_vec_dict[x])

            self.train_data = self.make_data_obj(train_data, train_vector, train_y)
            self.test_data = self.make_data_obj(test_data, test_vector, test_y)
            self.vocab = tokenizer.word_index
            self.vocabulary_size = len(self.vocab)
            self.embed_matrix =  self.build_char_embedding_matrix()


    def make_char_doc_vector(self, uniq_doc):

        pool = Pool(processes=4)
        uniq_doc_clean = pool.map(clean_text_major, uniq_doc)

        # create an tokenizer to convert document into numerical vector
        # notice we limit vocab size here
        tokenizer = Tokenizer(lower=False, char_level=True)
        tokenizer.fit_on_texts(uniq_doc_clean)
        uniq_seq = tokenizer.texts_to_sequences(uniq_doc_clean)
        uniq_seq = pad_sequences(uniq_seq, maxlen=self.target_doc_len,
                                 padding="post", truncating="post")

        # a map from raw doc to vec sequence
        return dict(zip(uniq_doc, uniq_seq)), tokenizer

    def make_word_doc_vector(self, uniq_doc):

        pool = Pool(processes=4)
        uniq_doc_clean = pool.map(clean_text_minor, uniq_doc)

        # create an tokenizer to convert document into numerical vector
        # notice we limit vocab size here
        tokenizer = Tokenizer(num_words=self.vocabulary_size)
        tokenizer.fit_on_texts(uniq_doc_clean)
        uniq_seq = tokenizer.texts_to_sequences(uniq_doc_clean)
        uniq_seq = pad_sequences(uniq_seq, maxlen=self.target_doc_len,
                                 padding="post", truncating="post")

        # a map from raw doc to vec sequence
        return dict(zip(uniq_doc, uniq_seq)), tokenizer

    def match_domain_combo(self, train_data):
        uniq_doc = pd.unique(train_data["k_doc"].values.ravel('K'))
        domain_problem_list = []
        for doc in uniq_doc:
            domain_rows = train_data.loc[train_data['k_doc'] == doc]
            domain_problem_list.append(domain_rows)
        return domain_problem_list

    def load_dataframe(self, path_to_pickle):
        data_pickle = Path(path_to_pickle)
        if not data_pickle.exists():
            logging.info("loading data structure from RAW")
            loader = PANData(self.year, train_split=self.train_split, test_split=self.test_split)
            train_data, test_data = loader.get_data()

            # Takes out label column
            train_y = train_data['label'].tolist()
            test_y = test_data['label'].tolist()
            train_data.drop(['label'], axis=1, inplace=True)
            test_data.drop(['label'], axis=1, inplace=True)
            train_y = np.array([1 if lbl == "Y" else 0 for lbl in train_y])
            test_y = np.array([1 if lbl == "Y" else 0 for lbl in test_y])

            logging.info("load data structure completed")

            # pickle.dump([train_data, test_data, train_y, test_y], open(data_pickle, mode="wb"))
            # logging.info("dumped all data structure in " + str(data_pickle))
        else:
            logging.info("loading data structure from PICKLE")
            [train_data, test_data, train_y, test_y] = pickle.load(data_pickle.open(mode="rb"))
            logging.info("load data structure completed")

        return (train_data, train_y), (test_data, test_y)

    def make_data_obj(self, data_raw, data_vector, labels):
        logging.info("data shape: " + str(data_vector.shape))
        logging.info("label shape: " + str(labels.shape))

        data_obj = DataObject(self.problem_name, len(labels))
        data_obj.raw = data_raw
        data_obj.value = data_vector
        data_obj.label_doc = labels
        return data_obj


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    a = DataBuilderPan(year="15", train_split="pan15_train", test_split="pan15_test",
                       embed_dim=100, vocab_size=30000, target_doc_len=10000, target_sent_len=1024)
