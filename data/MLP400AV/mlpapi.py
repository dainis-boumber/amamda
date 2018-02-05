import os
import csv
import random as rand
import sklearn.model_selection as select
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


# API container, static methods only
class MLPAPI:
    AUTHORS = (
        'geoffrey_hinton',
        'vapnik',
        'bernard_scholkopf',
        'thomas_l_griffiths',
        'yann_lecun',
        'xiaojin_zhu',
        'yee_whye_teh',
        'radford_neal',
        'david_blei',
        'alex_smola',
        'michael_jordan',
        'zoubin_ghahramani',
        'daphne_koller',
        'lawrence_saul',
        'trevor_hastie',
        'thorsten_joachims',
        'yoshua_bengio',
        'andrew_y_ng',
        'tom_mitchell',
        'robert_tibshirani'
    )

    # col indeces
    K_AUTHOR = 0
    K_PAPERS = 1
    U_AUTHOR = 2
    U_PAPERS = 3
    LABEL = 4

    author_ix = 0
    papers_ix = 1
    pa_table_ix = 2

    def __init__(self):
        pass

    @staticmethod
    def read_input(infile='labels.csv'):
        with open(infile) as fin:
            paper_authors = {}
            # for root, dirs, files in os.walk('./' + label):
            #    if root != '.':
            #        authors.append(str(root.lstrip('./')))
            csv_reader = csv.reader(fin, delimiter=',')
            authors = csv_reader.next()[1:]  # skip header
            papers = []

            for line in csv_reader:
                paper = line[0]
                authorship = line[1:]
                tmp = []
                for i, value in enumerate(authorship):
                    if value == '1':
                        tmp.append(authors[i])
                paper_authors[paper] = tmp
                papers.append(paper)

            assert (len(authors) == 20)
            assert (len(papers) == 400)
            assert (len(paper_authors) == 400)
            for paper in papers:
                assert len(paper_authors[paper]) >= 1, paper

            return authors, papers, paper_authors

    @staticmethod
    def write_pairs(filename, pairs):
        with open(filename, 'w') as of:
            of.write('Author,Known_paper,Unknown_paper,Is_same_author\n')
            for pair in pairs:
                author = pair[0]
                paper = pair[1][0]
                unknown = pair[1][1]
                label = pair[2]
                of.write(author + ',' + paper + ',unknown,' + unknown + ',' + label + '\n')

    @staticmethod
    def write_dataset(filenames, pair_sets):
        for i, filename in enumerate(filenames):
            MLPAPI.write_pairs(filename, pair_sets[i])

    @staticmethod
    def split_by_author(authors, papers, paper_authors):

        i = 0
        j = len(papers) / len(authors)
        bins = []

        while j <= len(papers):
            selection = papers[i:j]
            assert (len(selection) == len(papers) / len(authors))
            for s in selection:
                assert s is not None, selection

            for a in authors:
                assert a is not None, authors
                if a in selection[MLPAPI.papers_ix]:
                    author = a
                    break
            for i, s in enumerate(selection):
                assert author in s, author

            shuffled_papers = shuffle(selection)
            bins.append((author, shuffled_papers, paper_authors))
            i = j
            j += len(papers) / len(authors)

            assert len(bins[-1][MLPAPI.papers_ix]) == len(papers) / len(authors), str(bins[-1][MLPAPI.papers_ix])
            assert len(bins[-1][MLPAPI.pa_table_ix]) == len(paper_authors), str(bins[-1][MLPAPI.pa_table_ix])

        assert len(bins) == len(authors), len(bins)

        return shuffle(bins)

    @staticmethod
    def tr_tst_val_split(author_bins, ntr=14, ntst=4, nval=2):
        assert len(author_bins) == ntr + ntst + nval
        tr_bins = []
        tst_bins = []
        val_bins = []
        for i, bin in enumerate(author_bins):
            train, test = select.train_test_split(bin[MLPAPI.papers_ix], test_size=ntst)
            train, val = select.train_test_split(train, train_size=ntr)
            tr_bins.append((bin[MLPAPI.author_ix], train, bin[MLPAPI.papers_ix]))
            tst_bins.append((bin[MLPAPI.author_ix], test, bin[MLPAPI.papers_ix]))
            val_bins.append((bin[MLPAPI.author_ix], val, bin[MLPAPI.papers_ix]))
            assert len(tr_bins[i][MLPAPI.papers_ix]) == ntr, str(len(tr_bins[i][MLPAPI.papers_ix]))
            assert len(tst_bins[i][MLPAPI.papers_ix]) == ntst, str(len(tst_bins[i][MLPAPI.papers_ix]))
            assert len(val_bins[i][MLPAPI.papers_ix]) == nval, str(val_bins[i][MLPAPI.papers_ix])
        assert len(tr_bins) == len(author_bins)
        assert len(tst_bins) == len(author_bins)
        assert len(val_bins) == len(author_bins)
        return tr_bins, tst_bins, val_bins

    @staticmethod
    def make_pairs(label, yes_no, bins):
        if label is not 'YES' and label is not 'NO':
            raise ValueError('Label must always be YES or NO')

        X = []
        y = []
        pairs = []
        for unknown, author in yes_no:
            for b in bins:
                for paper in b[MLPAPI.papers_ix]:
                    if b[MLPAPI.author_ix] == author:
                        if label is 'NO':
                            assert unknown != paper
                        X.append([author, paper, unknown])
                        y.append(label)
                        pairs.append((author, (paper, unknown), label))
        return shuffle(pairs)

    @staticmethod
    def make_unkown_papers(author_bins, label, paper_authors, scheme='not A2'):
        bins = author_bins
        npapers_original = len(author_bins[0][MLPAPI.papers_ix])
        yes_no = []
        nneg = 0
        author_idx = {}
        for i, author in enumerate(MLPAPI.AUTHORS):
            for b in bins:
                if author == b[MLPAPI.author_ix]:
                    author_idx.update({author: i})

        for i, b in enumerate(bins):
            current_author = b[MLPAPI.author_ix]
            if label == 'YES':
                unknown = rand.choice(b[MLPAPI.papers_ix])
                assert unknown is not None, b[MLPAPI.author_ix]
                yes_no.append((unknown, current_author))
                nneg += 1
                if scheme != 'A2':
                    b[MLPAPI.papers_ix].remove(unknown)
            elif label == 'NO':
                possible_negatives = [author for author in MLPAPI.AUTHORS if author != current_author]
                neg_found = False
                unknown = None
                inegative_author = -1
                while not neg_found:
                    negative_author = rand.choice(possible_negatives)
                    inegative_author = author_idx[negative_author]
                    npapers = len(bins[inegative_author][MLPAPI.papers_ix])

                    if npapers_original - npapers == 0:
                        for j in range(len(bins[inegative_author][MLPAPI.papers_ix])):
                            unknown = rand.choice(bins[inegative_author][MLPAPI.papers_ix])
                            co_authors = paper_authors[unknown]
                            if current_author not in co_authors:
                                neg_found = True
                                break
                            if j == len(bins[inegative_author][MLPAPI.papers_ix]) - 1:
                                possible_negatives.remove(negative_author)
                                if len(possible_negatives) == 0:
                                    raise RuntimeError(
                                        'all papers sampled in co authoship. Just bad luck, try running again')
                    else:
                        possible_negatives.remove(negative_author)
                assert unknown is not None
                yes_no.append((unknown, current_author))
                nneg += 1
                if scheme != 'A2':
                    bins[inegative_author][MLPAPI.papers_ix].remove(unknown)
            else:
                raise ValueError('label is always either YES or NO')

        for i, b in enumerate(bins):
            print("Bins length: " + str(len(bins[i][MLPAPI.papers_ix])) + " b length: " + str(len(b[MLPAPI.papers_ix])))

        assert nneg == len(author_bins), nneg
        return yes_no, bins

    @staticmethod
    def create_dataset(dir, scheme):
        labels = ['YES', 'NO']
        pairs = [[], [], []]
        all_pairs = []
        fnames = (dir + '/' + scheme + 'train.csv', dir + '/' + scheme + 'val.csv', dir + '/' + scheme + 'test.csv')

        for i, label in enumerate(labels):
            authors, papers, paper_authors = MLPAPI.read_input()
            author_bins = MLPAPI.split_by_author(authors, papers, paper_authors)
            if scheme == 'A' or scheme == 'A2':  # test, train and val do NOT intersect
                tr_author_bins, tst_author_bins, val_author_bins = MLPAPI.tr_tst_val_split(author_bins)
                for j, bins in enumerate((tr_author_bins, val_author_bins, tst_author_bins)):
                    yes_no, bins_sans_unknown = MLPAPI.make_unkown_papers(author_bins=bins, label=label,
                                                                          paper_authors=paper_authors, scheme=scheme)
                    pairs[j].extend(MLPAPI.make_pairs(label, yes_no=yes_no, bins=bins_sans_unknown))
                    shuffle(pairs[j])
            elif scheme == 'B':  # test train and val MAY intersect
                yes_no, bins_sans_unknown = MLPAPI.make_unkown_papers(author_bins=author_bins, label=label,
                                                                      paper_authors=paper_authors, scheme=scheme)
                all_pairs.extend(MLPAPI.make_pairs(label, yes_no=yes_no, bins=bins_sans_unknown))
                if i == len(labels) - 1:
                    shuffle(all_pairs)
                    train, test = select.train_test_split(all_pairs, train_size=0.8)
                    train, val = select.train_test_split(train, test_size=int(len(test) / 2))
                    pairs = [train, val, test]
            else:
                raise ValueError

        MLPAPI.write_dataset(fnames, pairs)
        return pairs

    @staticmethod
    def load_dataset(fileformat='lines', scheme='A2', directory=os.path.join(os.path.dirname(__file__), ''),
                     path_train='train.csv', path_test='test.csv', path_val='val.csv'):
        train_test_val_paths = (directory + '/' + scheme + path_train,  # ex.: /home/user/A2train.csv
                                directory + '/' + scheme + path_test,
                                directory + '/' + scheme + path_val)
        train = []
        test = []
        val = []
        for path in train_test_val_paths:
            with open(path) as pt:
                reader = csv.reader(pt)
                next(reader)
                for row in reader:
                    label = row[MLPAPI.LABEL]
                    kauthor = row[MLPAPI.K_AUTHOR]
                    try:
                        with open(directory + '/' + label + '/' + row[MLPAPI.K_AUTHOR] + '/' + \
                                          row[MLPAPI.K_PAPERS], 'r') as f:
                            if fileformat == 'lines':
                                k = f.readlines()
                            elif fileformat == 'str' or fileformat == 'pandas':
                                k = f.read()
                            else:
                                raise AttributeError

                        # label the unknown file using author string in the file name
                        author = None
                        for author in MLPAPI.AUTHORS:
                            if author in row[MLPAPI.U_PAPERS]:
                                break

                        with open(directory + '/' + label + '/' + author + '/' + row[MLPAPI.U_PAPERS], 'r') as f:
                            if fileformat == 'lines':
                                u = f.readlines()
                            elif fileformat == 'str' or fileformat == 'pandas':
                                u = f.read()
                            else:
                                raise AttributeError
                    except UnicodeDecodeError:
                        print(directory + '/' + label + '/' + row[MLPAPI.K_AUTHOR] + '/' + row[MLPAPI.K_PAPERS], 'r')
                        raise UnicodeDecodeError

                    if 'train' in path:
                        train.append({'k_author': kauthor, 'k_doc': k, 'u_author': author, 'u_doc': u, 'label': label})
                    elif 'test' in path:
                        test.append({'k_author': kauthor, 'k_doc': k, 'u_author': author, 'u_doc': u, 'label': label})
                    elif 'val' in path:
                        val.append({'k_author': kauthor, 'k_doc': k, 'u_author': author, 'u_doc': u, 'label': label})
                    else:
                        raise ValueError
        train = shuffle(train, random_state=42)
        test = shuffle(test, random_state=42)
        val = shuffle(val, random_state=42)
        if fileformat == 'pandas':
            return pd.DataFrame(train), pd.DataFrame(val), pd.DataFrame(test)
        return train, val, test


class MLPVLoader:
    def __init__(self, scheme='A2', fileformat='pandas', directory=None):
        if directory is None:
            directory = os.path.join(os.path.dirname(__file__), '')
        self.train, self.val, self.test = MLPAPI.load_dataset(fileformat=fileformat, scheme=scheme, directory=directory)

    def get_mlpv(self):
        return self.train, self.val, self.test

    def slice(self, l, n):
        return np.array_split(l, n)

    def get_slices(self, n):
        return self.slice(self.train, n), self.slice(self.val, n), self.slice(self.test, n)


def main():
    # MLPAPI.create_dataset(scheme='B')
    loader = MLPVLoader()
    tr, v, tst = loader.get_slices(5)
    print('done')


if __name__ == "__main__":
    main()
