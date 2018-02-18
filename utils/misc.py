# -*- coding: utf-8 -*-
import datetime
import logging
import os


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


def get_date():
    date_today = datetime.datetime.now().strftime("%y%m%d")
    return date_today


def get_time():
    date_today = datetime.datetime.now().strftime("%Y-%b-%d %H:%M:%S")
    return date_today


def get_exp_logger(am):
    log_path = am.get_exp_log_path()
    # logging facility, log both into file and console
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_path,
                        filemode='w+')
    console_logger = logging.StreamHandler()
    logging.getLogger('').addHandler(console_logger)
    logging.info("log created: " + log_path)
