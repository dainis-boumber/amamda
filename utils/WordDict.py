# -*- coding: utf-8 -*-

import os

from collections import Counter


def dictFromFile(aFileName):
    with open(aFileName) as inFile:
        listsOfWords = inFile.read().split()
    return Counter(listsOfWords)


def getListListPANAndFoldersPAN(aPath):
    split_name = []
    split_instance_list = []

    for d in os.listdir(aPath):
        aListPAN = []
        if os.path.isfile(aPath + "/" + d):
            continue
        split_name.append(d)
        for f in os.listdir(aPath + "/" + d + "/"):
            if os.path.isfile(aPath + "/" + d + "/" + f):
                continue
            aListPAN.append(aPath + "/" + d + "/" + f)
        split_instance_list.append(sorted(aListPAN))
    split_name, split_instance_list = zip(*sorted(zip(split_name, split_instance_list)))

    return split_instance_list, split_name
