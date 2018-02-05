import os
from utils.WordDict import getListListPANAndFoldersPAN


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


class PANData:
    def __init__(self, year):
        p = os.path.abspath(__file__ + "/../../data/PAN" + str(year) + '/')
        pair_dirs, split_names = getListListPANAndFoldersPAN(p)
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


