import time
import os
from misc import get_date


class ArchiveManager(object):

    def __init__(self, data_name, input_name, middle_name, output_name, truth_file=None):
        self.data_name = data_name
        self.input_name = input_name
        self.middle_name = middle_name
        self.output_name = output_name
        self.truth_file = truth_file
        self.time_stamp = str(int(time.time()))

    def get_tag(self):
        tag = self.data_name + "_" + self.input_name + "_" + self.middle_name + "_" + self.output_name
        return tag

    def get_tag_dir(self):
        path = os.path.join(".", "runs", self.get_tag(), "")
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_exp_dir(self):
        date_today = get_date()
        if self.truth_file is None:
            path = os.path.join(self.get_tag_dir(), date_today + "_" + self.time_stamp, "")
        else:
            path = os.path.join(self.get_tag_dir(), date_today + "_" + self.time_stamp + "_" + self.truth_file, "")
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_exp_log_path(self):
        return os.path.join(self.get_exp_dir(), "log.txt")


if __name__ == '__main__':
    am = ArchiveManager("ML", "test", "test", "test")
    print((am.get_tag()))
    print((am.get_tag_dir()))
    print((am.get_exp_dir()))
    print((am.get_exp_log_path()))
