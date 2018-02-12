import numpy as np
import logging
import os

from data_helper.ds_models import PANData
from data_helper.DataHelpers import DataHelper
from data_helper.Data import DataObject


class DataBuilderPan(DataHelper):

    problem_name = "PAN"

    sent_num_file = ["aspect_0.count", "test_aspect_0.count"]
    rating_file = ["aspect_0.rating", "test_aspect_0.rating"]
    content_file = ["aspect_0.txt", "test_aspect_0.txt"]

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

        self.dataset_dir = self.data_path + 'MLP400AV/'
        self.num_classes = 2  # true or false

        self.load_all_data()

    def get_dir_list(self, dataset_dir):
        split_name = []
        split_dir_list = []

        for d in os.listdir(dataset_dir):
            problem_dir_list = []
            split_dir = os.path.join(dataset_dir, d)
            if os.path.isfile(split_dir):
                continue
            split_name.append(d)
            for problem in os.listdir(split_dir):
                problem_dir = os.path.join(split_dir, problem)
                if os.path.isfile(problem_dir):
                    continue
                problem_dir_list.append(problem_dir)
            split_dir_list.append(sorted(problem_dir_list))
        result = dict(zip(split_name, split_dir_list))

        return result

    def dir_loader(self, year, train_split, test_split):
        p = os.path.abspath(__file__ + "/../../data/PAN" + str(year) + '/')
        dir_list = self.get_dir_list(p)
        labels = []
        self.splits = []
        self.name = 'PAN' + str(year)
        for i, split_name in enumerate(split_names):
            if 'train' in split_name:
                with open(os.path.join(p, split_name, 'truth.txt')) as truth:
                    for line in truth:
                        labels.append(line.strip().split()[1])
                    self.splits.append(Split(split_name, pair_dirs[i], labels))
            else:
                self.splits.append(Split(split_name, pair_dirs[i], None))

    def load_files(self):

        data = PANData(self.year)

        raise NotImplementedError

        if self.doc_as_sent:
            x_text = DataBuilderPan.concat_to_doc(sent_list=x_text, sent_count=sent_count)

        x = []
        for train_line_index in range(len(x_text)):
            tokens = x_text[train_line_index].split()
            x.append(tokens)

        data = DataObject(self.problem_name, len(y))
        data.raw = x
        data.label_doc = y_onehot
        data.doc_size = sent_count

        return data

    def to_list_of_sent(self, sentence_data, sentence_count):
        x = []
        index = 0
        for sc in sentence_count:
            one_review = sentence_data[index:index+sc]
            x.append(one_review)
            index += sc
        return np.array(x)

    def load_all_data(self):
        train_data = self.load_files()
        self.vocab, self.vocab_inv = self.build_vocab([train_data], self.vocabulary_size)
        self.embed_matrix = self.build_glove_embedding(self.vocab_inv)
        train_data = self.build_content_vector(train_data)
        train_data = self.pad_sentences(train_data)

        if self.doc_level:
            value = self.to_list_of_sent(train_data.value, train_data.doc_size)
            train_data.value = value
            DataHelper.pad_document(train_data, self.target_doc_len)

        self.train_data = train_data
        self.train_data.embed_matrix = self.embed_matrix
        self.train_data.vocab = self.vocab
        self.train_data.vocab_inv = self.vocab_inv
        self.train_data.label_instance = self.train_data.label_doc

        test_data = self.load_files(1)
        test_data = self.build_content_vector(test_data)
        test_data = self.pad_sentences(test_data)

        if self.doc_level:
            value = self.to_list_of_sent(test_data.value, test_data.doc_size)
            test_data.value = value
            DataHelper.pad_document(test_data, self.target_doc_len)

        self.test_data = test_data
        self.test_data.embed_matrix = self.embed_matrix
        self.test_data.vocab = self.vocab
        self.test_data.vocab_inv = self.vocab_inv
        self.test_data.label_instance = self.test_data.label_doc

    def load_dataframe(self):
        data_pickle = Path("av400tuple.pickle")
        if not data_pickle.exists():

        else:
            logging.info("loading data structure from PICKLE")
            [train_data, val_data, test_data, train_y, val_y, test_y] = pickle.load(open(data_pickle, mode="rb"))
            logging.info("load data structure completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    a = DataBuilderPan(embed_dim=300, target_doc_len=64, target_sent_len=1024, aspect_id=None,
                       doc_as_sent=False, doc_level=True)
