import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from base import DataObject, PANData, DataBuilder, clean_text


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
        self.load_and_proc_data()

    def load_and_proc_data(self):
        (train_data, train_y), (test_data, test_y) = self.load_dataframe()

        all_data = pd.concat([train_data, test_data])
        uniq_doc = pd.unique(all_data.values.ravel('K'))

        pool = Pool(processes=4)
        uniq_doc_clean = pool.map(clean_text, uniq_doc)

        # doc_lens = [len(d) for d in uniq_doc]
        # print( sorted(doc_lens, reverse=True)[:20] )

        # notice we limit vocab size here
        tokenizer = Tokenizer(num_words=self.vocabulary_size)
        tokenizer.fit_on_texts(uniq_doc_clean)
        uniq_seq = tokenizer.texts_to_sequences(uniq_doc_clean)
        uniq_seq = pad_sequences(uniq_seq, maxlen=self.target_doc_len,
                                 padding="post", truncating="post")

        # a map from raw doc to vec sequence
        raw_to_vec = dict(zip(uniq_doc, uniq_seq))

        self.train_data = self.proc_data(train_data, train_y, raw_to_vec)
        self.test_data = self.proc_data(test_data, test_y, raw_to_vec)

        self.domain_list = self.match_domain_combo(train_data)

        self.vocab = tokenizer.word_index
        self.embed_matrix =  self.build_embedding_matrix

    def match_domain_combo(self, train_data):
        uniq_doc = pd.unique(train_data["k_doc"].values.ravel('K'))
        domain_problem_list = []
        for doc in uniq_doc:
            domain_rows = train_data.loc[train_data['k_doc'] == doc]
            domain_problem_list.append(domain_rows)
        return domain_problem_list

    def load_dataframe(self):
        data_pickle = Path("PAN15tuple.pickle")
        if not data_pickle.exists():
            logging.info("loading data structure from RAW")
            loader = PANData(self.year, train_split=self.train_split, test_split=self.test_split)
            train_data, test_data = loader.get_data

            train_y = train_data['label'].tolist()
            test_y = test_data['label'].tolist()

            train_data.drop(['label'], axis=1, inplace=True)
            test_data.drop(['label'], axis=1, inplace=True)

            logging.info("load data structure completed")

            # pickle.dump([train_data, test_data, train_y, test_y], open(data_pickle, mode="wb"))
            # logging.info("dumped all data structure in " + str(data_pickle))
        else:
            logging.info("loading data structure from PICKLE")
            [train_data, test_data, train_y, test_y] = pickle.load(open(data_pickle, mode="rb"))
            logging.info("load data structure completed")

        return (train_data, train_y), (test_data, test_y)

    def proc_data(self, data_raw, label, raw_to_vec):
        vector_sequences = data_raw.applymap(lambda x: raw_to_vec[x])
        doc_label = np.array([1 if lbl == "Y" else 0 for lbl in label])

        logging.info("data shape: " + str(vector_sequences.shape))
        logging.info("label shape: " + str(doc_label.shape))

        if self.sent_split:
            raise NotImplementedError

        data_obj = DataObject(self.problem_name, len(doc_label))
        data_obj.raw = data_raw
        data_obj.label_doc = doc_label
        data_obj.value = vector_sequences
        return data_obj


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    a = DataBuilderPan(year="15", train_split="pan15_train", test_split="pan15_test",
                       embed_dim=100, vocab_size=30000, target_doc_len=10000, target_sent_len=1024)

    pass
