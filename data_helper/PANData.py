import os
import pandas as pd
from utils.WordDict import get_dir_list


class Pair:
    def __init__(self, full_pair_dir_path, label):
        self.unknown = None
        self.known = []
        _, tail = os.path.split(full_pair_dir_path)
        self.name = tail
        self.label = label
        for path in os.listdir(full_pair_dir_path):
            if 'unknown' in path:
                self.unknown = path
            else:
                self.known.append(path)


class Split:
    def __init__(self, name, full_pair_dir_paths, labels=None):
        self.name = name
        self.pairs = []
        for i, path in enumerate(full_pair_dir_paths):
            if labels is not None:
                self.pairs.append(Pair(path, labels[i]))
            else:
                self.pairs.append(Pair(path, None))


class PANData(object):
    def __init__(self, year, train_split, test_split):
        p = os.path.abspath(__file__ + "/../../data/PAN" + str(year) + '/')
        self.year = year
        self.name = 'PAN' + str(year)

        assert os.path.exists(os.path.join(p, train_split))
        assert os.path.exists(os.path.join(p, test_split))

        dir_list = get_dir_list(p)

        train_labels = []
        with open(os.path.join(p, train_split, 'truth.txt')) as truth_file:
            for line in truth_file:
                train_labels.append(line.strip().split())
        train_labels = dict(train_labels)

        test_labels = []
        with open(os.path.join(p, test_split, 'truth.txt')) as truth_file:
            for line in truth_file:
                test_labels.append(line.strip().split())
        test_labels = dict(test_labels)

        self.train_splits = []
        for problem_dir in dir_list[train_split]:
            k, u = self.load_one_problem(problem_dir)
            l = train_labels[os.path.basename(problem_dir)]
            self.train_splits.append({'k_doc': k, 'u_doc': u, "label": l})

        self.test_splits = []
        for problem_dir in dir_list[test_split]:
            k, u = self.load_one_problem(problem_dir)
            l = test_labels[os.path.basename(problem_dir)]
            self.test_splits.append({'k_doc': k, 'u_doc': u, "label": l})

        self.train_splits = pd.DataFrame(self.train_splits)
        self.test_splits = pd.DataFrame(self.test_splits)

    def get_data(self):
        return self.train_splits, self.test_splits

    def get_train(self):
        return self.train_splits

    def get_test(self):
        return self.test_splits

    def get_train_domains(self):
        return self.train_domains

    def get_test_domains(self):
        return self.test_domains

    def get_domains(self):
        return self.train_domains, self.test_domains

    @staticmethod
    def load_one_problem(problem_dir):
        doc_file_list = os.listdir(problem_dir)
        u = None
        k = None
        if len(doc_file_list) > 2:
            print(problem_dir + " have more " + str(len(doc_file_list)) + " files!")
        for doc_file in doc_file_list:
            with open(os.path.join(problem_dir, doc_file)) as f:
                if doc_file.startswith("known"):
                    k = f.read()
                elif doc_file.startswith("unknown"):
                    u = f.read()
                else:
                    print(doc_file + " is not right!")
        return k, u


if __name__ == "__main__":
    dater = PANData("15", train_split="pan15_train", test_split="pan15_test")
    print("t")


