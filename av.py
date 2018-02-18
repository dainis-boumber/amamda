
if __name__ == "__main__":
    raise NotImplementedError

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


    start = timer()
    # n_fc variable controls how many fc layers you got at the end, n_conv does that for conv layers

    tt.training(filter_sizes=[[3, 4, 5]], num_filters=100, l2_lambda=0, dropout_keep_prob=0.75,
                batch_normalize=False, elu=False, fc=[], n_steps=5000)
    end = timer()
    print((end - start))

    # ev.evaluate(am.get_exp_dir(), None, doc_acc=True, do_is_training=True)
