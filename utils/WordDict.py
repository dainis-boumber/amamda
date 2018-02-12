# -*- coding: utf-8 -*-

import os

from collections import Counter


def dictFromFile(aFileName):
    with open(aFileName) as inFile:
        listsOfWords = inFile.read().split()
    return Counter(listsOfWords)


def get_dir_list(dataset_dir):
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
