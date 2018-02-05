from enum import Enum
import logging

from data_helper.Data import DataObject
from nn.input_components.OneDocSequence import OneDocSequence
from nn.input_components.OneSequence import OneSequence
from nn.middle_components.DocumentCNN import DocumentCNN
from nn.middle_components.SentenceCNN import SentenceCNN
from nn.output_components.Output import Output


class OutputNNType(Enum):
    OriginOutput = 0
    LSAAC1Output = 1
    LSAAC1_INDIBIAS_Output = 2
    LSAAC1_MASK_Output = 3
    LSAAC2Output = 4
    LSAAR1Output = 5
    LSAAR1Output_SentFCOverall = 6
    LSAAR1Output_ShareScore = 7
    AAAB = 100


class CNNNetworkBuilder:
    """I"m currently calling this CNN builder because i'm not sure if it can handle future
    RNN parameters, and just for flexibility and ease of management the component maker is being made into
    separate function
    """
    def __init__(self, input_comp, middle_comp, output_comp):

        # input component =====
        self.input_comp = input_comp

        self.input_x = self.input_comp.input_x
        self.input_y = self.input_comp.input_y
        self.dropout_keep_prob = self.input_comp.dropout_keep_prob

        # middle component =====
        self.middle_comp = middle_comp

        # output component =====
        self.output_comp = output_comp

        self.scores = self.output_comp.scores
        self.predictions = self.output_comp.predictions
        self.loss = self.output_comp.loss
        self.accuracy = self.output_comp.accuracy

    @staticmethod
    def get_input_component(input_name, data):
        # input component =====
        if input_name == "Sentence":
            input_comp = OneSequence(data)
        elif input_name == "Document":
            input_comp = OneDocSequence(data=data)
        else:
            raise NotImplementedError

        return input_comp

    @staticmethod
    def get_middle_component(middle_name, input_comp, data,
                             filter_size_lists=None, num_filters=None, dropout=None,
                             batch_norm=None, elu=None, fc=[], l2_reg=0.0):
        logging.info("setting: %s is %s", "filter_size_lists", filter_size_lists)
        logging.info("setting: %s is %s", "num_filters", num_filters)
        logging.info("setting: %s is %s", "batch_norm", batch_norm)
        logging.info("setting: %s is %s", "elu", elu)
        logging.info("setting: %s is %s", "MIDDLE_FC", fc)

        if middle_name == 'Origin':
            middle_comp = SentenceCNN(previous_component=input_comp, data=data,
                                      filter_size_lists=filter_size_lists, num_filters=num_filters,
                                      dropout=dropout, batch_normalize=batch_norm, elu=elu,
                                      fc=fc, l2_reg_lambda=l2_reg)
        elif middle_name == "DocumentCNN":
            middle_comp = DocumentCNN(previous_component=input_comp, data=data,
                                      filter_size_lists=filter_size_lists, num_filters=num_filters,
                                      dropout=dropout, batch_normalize=batch_norm, elu=elu,
                                      fc=fc, l2_reg_lambda=l2_reg)
        else:
            raise NotImplementedError

        return middle_comp

    @staticmethod
    def get_output_component(output_name, middle_comp, data, l2_reg=0.0):
        if "??" in output_name:
            output_comp = Output(middle_comp, data=data, l2_reg=l2_reg)
        else:
            raise NotImplementedError

        return output_comp


if __name__ == "__main__":
    data = DataObject("test", 100)
    data.vocab = [1, 2, 3]