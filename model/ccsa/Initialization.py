import numpy as np
import keras
import logging
import sys
import pandas as pd

from data.base import DataObject

from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.models import Model
from keras.layers import Flatten, Dense
from keras import backend as K
from keras.layers import Embedding


def printn(string):
    sys.stdout.write(string)
    sys.stdout.flush()


def euclidean_distance(vects):
    eps = 1e-08
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), eps))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def dg_cnn(data_builder):
    k_input = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="k_input")
    u_input = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="u_input")

    embedding_layer = Embedding(input_length=data_builder.target_doc_len,
                                input_dim=data_builder.vocabulary_size + 1,
                                output_dim=100,
                                weights=[data_builder.embed_matrix],
                                trainable=False)

    k_embedded_seq = embedding_layer(k_input)
    u_embedded_seq = embedding_layer(u_input)

    conv_first = Conv1D(filters=128, kernel_size=5, activation='relu')
    poll_first = MaxPooling1D(pool_size=1024)

    k_cov = conv_first(k_embedded_seq)
    k_poll = poll_first(k_cov)

    u_cov = conv_first(u_embedded_seq)
    u_poll = poll_first(u_cov)

    x = keras.layers.subtract([k_poll, u_poll])

    x = Flatten()(x)
    diff_embedding = Dense(128, activation='relu')(x)

    model = Model([k_input, u_input], diff_embedding)

    return model


def load_data(data_builder):
    train = data_builder.train_data
    test = data_builder.test_data
    pair_combo, y1, y2, y_combo = make_pairs(train)
    return (pair_combo, y1, y2, y_combo), (test.value, test.label_doc)


def make_pairs(data_object: DataObject):
    source_pair = []
    target_pair = []
    source_l = []
    target_l = []
    value_frame = data_object.value
    label_frame = data_object.label_doc
    for trs in range(len(value_frame)):
        for trt in range(trs + 1, len(value_frame)):
            source_pair.append(value_frame.iloc[[trs]])
            target_pair.append(value_frame.iloc[[trt]])
            source_l.append(label_frame[trs])
            target_l.append(label_frame[trt])

    source_pair = pd.concat(source_pair, axis=0)
    target_pair = pd.concat(target_pair, axis=0)
    source_l = np.array(source_l)
    target_l = np.array(target_l)

    source_pair.columns = ["s_k_doc", "s_u_doc"]
    target_pair.columns = ["t_k_doc", "t_u_doc"]
    source_pair = source_pair.reset_index(drop=True)
    target_pair = target_pair.reset_index(drop=True)
    pair_combo = source_pair.join(target_pair, how='outer')

    y_combo = source_l == target_l
    y_combo = y_combo.astype(int)

    return pair_combo, source_l, target_l, y_combo

def training_the_model(model, train, test, epochs=80, batch_size=256):
    pair_combo, y1, y2, y_combo = train
    test_value, test_label = test

    print('Training the model - Epochs '+str(epochs))
    best_acc = 0
    if batch_size > len(y2):
        print('Lowering batch size, to %d, number of inputs is too small for it.' % len(y2))
        batch_size = len(y2)
    for e in range(epochs):
        printn(str(e) + '->')
        for i in range(len(y2) // batch_size):
            # flipping stuff here
            from_sample = i * batch_size
            to_sample = (i + 1) * batch_size
            loss = model.train_on_batch([
                np.array(pair_combo["s_k_doc"].iloc[from_sample:to_sample].tolist()),
                np.array(pair_combo["s_u_doc"].iloc[from_sample:to_sample].tolist()),
                np.array(pair_combo["t_k_doc"].iloc[from_sample:to_sample].tolist()),
                np.array(pair_combo["t_u_doc"].iloc[from_sample:to_sample].tolist())
            ],
                [ y1[from_sample:to_sample], y_combo[from_sample:to_sample] ])

            loss = model.train_on_batch([
                np.array(pair_combo["t_k_doc"].iloc[from_sample:to_sample].tolist()),
                np.array(pair_combo["t_u_doc"].iloc[from_sample:to_sample].tolist()),
                np.array(pair_combo["s_k_doc"].iloc[from_sample:to_sample].tolist()),
                np.array(pair_combo["s_u_doc"].iloc[from_sample:to_sample].tolist()),
            ],
                [ y2[from_sample:to_sample], y_combo[from_sample:to_sample] ])


    Out = model.predict([[test_value["k_doc"], test_value["k_doc"]], [test_value["k_doc"], test_value["k_doc"]] ])
    Acc_v = np.array(Out == test_label)
    acc = (len(Acc_v) - np.count_nonzero(Acc_v) + .0000001) / len(Acc_v)
    logging.info("ACCU: " + str(acc))
    if best_acc < acc:
        best_acc = acc
        logging.info("BEST ACCU: " + str(acc))

    return best_acc
