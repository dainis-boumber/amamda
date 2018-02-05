import logging
from timeit import default_timer as timer

from data_helper.ds_models import PANData
from nn.TrainTask import TrainTask
from utils.ArchiveManager import ArchiveManager
from nn.CNNNetworkBuilder import CNNNetworkBuilder


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


if __name__ == "__main__":

    ###############################################
    # exp_names you can choose from at this point:
    #
    # Input Components:
    #
    # * TripAdvisor
    #
    # Middle Components:
    #
    # * Origin
    #
    #
    ################################################

    data_name = "PAN14"
    input_comp_name = "Document"
    middle_comp_name = "DocumentCNN"
    output_comp_name = "LSAA"

    am = ArchiveManager(data_name=data_name, input_name=input_comp_name, middle_name=middle_comp_name,
                        output_name=output_comp_name)
    get_exp_logger(am)
    logging.warning('===================================================')
    logging.debug("Loading data...")

    if data_name == "PAN13":
        dater = PANData('13')
    elif data_name == "PAN14":
        dater = PANData('14')
    elif data_name == "PAN15":
        dater = PANData('15')
    else:
        raise NotImplementedError

    input_comp = CNNNetworkBuilder.get_input_component(input_name=input_comp_name, data=dater)
    middle_comp = CNNNetworkBuilder.get_middle_component(middle_name=middle_comp_name, input_comp=input_comp,
                                                         data=dater,
                                                         filter_size_lists=[], num_filters=100,
                                                         dropout=0.0, batch_norm=None, elu=None, fc=[],
                                                         l2_reg=0.0)
    output_comp = CNNNetworkBuilder.get_output_component(output_name=output_comp_name, middle_comp=middle_comp,
                                                         data=dater, l2_reg=0.0)

    tt = TrainTask(data_helper=dater, am=am,
                   input_component=input_comp,
                   middle_component=middle_comp,
                   output_component=output_comp,
                   batch_size=32, evaluate_every=500, checkpoint_every=5000, max_to_keep=6,
                   restore_path=None)

    start = timer()
    # n_fc variable controls how many fc layers you got at the end, n_conv does that for conv layers

    tt.training(filter_sizes=[[3, 4, 5]], num_filters=100, l2_lambda=0, dropout_keep_prob=0.75,
                batch_normalize=False, elu=False, fc=[], n_steps=5000)
    end = timer()
    print((end - start))

    # ev.evaluate(am.get_exp_dir(), None, doc_acc=True, do_is_training=True)
