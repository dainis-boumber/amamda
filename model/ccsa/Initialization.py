import numpy as np
import keras
import logging
import sys
from pathlib import Path
from data_helper.DataBuilderML400 import DataBuilderML400
from data_helper.ds_models import PANData
from data_helper.DataHelpers import DataHelper
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten, Dense
from keras import backend as K


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

def dg_cnn(data_builder: DataBuilderML400):
    embedding_layer = Embedding(input_length=data_builder.target_doc_len,
                                input_dim=data_builder.vocabulary_size + 1,
                                output_dim=100,
                                weights=[data_builder.embed_matrix],
                                trainable=False)

    k_input = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="k_doc_input")
    k_embedded_seq = embedding_layer(k_input)
    u_input = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="u_doc_input")
    u_embedded_seq = embedding_layer(u_input)

    # shared first conv
    conv_first = Conv1D(filters=128, kernel_size=5, activation='relu')
    poll_first = MaxPooling1D(pool_size=1024)

    k_cov = conv_first(k_embedded_seq)
    k_poll = poll_first(k_cov)

    u_cov = conv_first(u_embedded_seq)
    u_poll = poll_first(u_cov)

    x = keras.layers.subtract([k_poll, u_poll])

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    model = Model([k_input, u_input], preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    return model

def load_data(input_dim=100):
    pan_data = PANData(15, 'pan15_train', 'pan15_test')
    train_domains = pan_data.get_train_domains()
    test_gomains = pan_data.get_test_domains()
    tr_pairs = []
    for i in range(train_domains):
        for j in range(train_domains):
            tr_pairs.append(make_pairs(train_domains[i], train_domains[j], input_dim=input_dim))
    test_pairs = []
    for i in range(test_gomains):
        for j in range(test_gomains):
            test_pairs.append(make_pairs(test_gomains[i], test_gomains[j], input_dim=input_dim))
    return tr_pairs, test_pairs

def make_pairs(source_domain, target_domain, input_dim):
    Training = []

    for trs in range(len(source_domain)):
        for trt in range(len(target_domain)):
            Training.append([trs, trt])

    X1k = np.zeros([len(Training), input_dim], dtype='float32')
    X1u = np.zeros([len(Training), input_dim], dtype='float32')
    y1 = np.zeros([len(Training)])
    X2k = np.zeros([len(Training), input_dim], dtype='float32')
    X2u = np.zeros([len(Training), input_dim], dtype='float32')
    y2 = np.zeros([len(Training)])
    yc = np.zeros([len(Training)])

    for i in range(len(Training)):
        in1, in2 = Training[i]
        X1k[i, :] = source_domain[in1].k_doc
        X1u[i, :] = source_domain[in1].u_doc
        X2k[i, :] = target_domain[in2].k_doc
        X2u[i, :] = target_domain[in2].u_doc
        y1[i] = source_domain[in1].label
        y2[i] = target_domain[in2].label

        if source_domain[in1].label == target_domain[in2].label:
            yc[i] = 1

    return (X1k, X1u, y1, X2k, X2u, y2, yc)

def training_the_model(model, X1k, X1u, y1, X2k, X2u, y2, yc, X_test_k, X_test_u, y_test, epochs=80, batch_size=256):

    print('Training the model - Epochs '+str(epochs))
    best_acc = 0
    if batch_size > len(y2):
        print('Lowering batch size, to %d, number of inputs is too small for it.' % len(y2))
        batch_size = len(y2)
    for e in range(epochs):
        if e % 10 == 0:
            printn(str(e) + '->')
        for i in range(len(y2) // batch_size):
            #flipping stuff here
            from_sample = i * batch_size
            to_sample = (i + 1) * batch_size
            loss = model.train_on_batch([[X1k[from_sample:to_sample, :, :],
                                         X1u[from_sample:to_sample, :, :]],
                                         [X2k[from_sample:to_sample, :, :],
                                         X2u[from_sample:to_sample, :, :]]],
                                        [y1[from_sample:to_sample, ],
                                         yc[from_sample:to_sample, ]])

            loss = model.train_on_batch([[X1u[from_sample:to_sample, :, :],
                                          X1k[from_sample:to_sample, :, :]],
                                         [X2u[from_sample:to_sample, :, :],
                                          X2k[from_sample:to_sample, :, :]]],
                                        [y1[from_sample:to_sample, ],
                                         yc[from_sample:to_sample, ]])

            loss = model.train_on_batch([[X1k[from_sample:to_sample, :, :],
                                          X1u[from_sample:to_sample, :, :]],
                                         [X2u[from_sample:to_sample, :, :],
                                          X2k[from_sample:to_sample, :, :]]],
                                        [y1[from_sample:to_sample, ],
                                         yc[from_sample:to_sample, ]])

            loss = model.train_on_batch([[X1u[from_sample:to_sample, :, :],
                                          X1k[from_sample:to_sample, :, :]],
                                         [X2k[from_sample:to_sample, :, :],
                                          X2u[from_sample:to_sample, :, :]]],
                                        [y1[from_sample:to_sample, ],
                                         yc[from_sample:to_sample, ]])

            loss = model.train_on_batch([[X2k[from_sample:to_sample, :, :, ],
                                         X2u[from_sample:to_sample, :, :, ]],
                                         [X1k[from_sample:to_sample, :, :, ],
                                         X1u[from_sample:to_sample, :, :, ]]],
                                        [y2[from_sample:to_sample, ],
                                         yc[from_sample:to_sample, ]])

            loss = model.train_on_batch([[X2u[from_sample:to_sample, :, :, ],
                                          X2k[from_sample:to_sample, :, :, ]],
                                         [X1u[from_sample:to_sample, :, :, ],
                                          X1k[from_sample:to_sample, :, :, ]]],
                                        [y2[from_sample:to_sample, ],
                                         yc[from_sample:to_sample, ]])
            loss = model.train_on_batch([[X2k[from_sample:to_sample, :, :, ],
                                          X2u[from_sample:to_sample, :, :, ]],
                                         [X1u[from_sample:to_sample, :, :, ],
                                          X1k[from_sample:to_sample, :, :, ]]],
                                        [y2[from_sample:to_sample, ],
                                         yc[from_sample:to_sample, ]])

            loss = model.train_on_batch([[X2u[from_sample:to_sample, :, :, ],
                                          X2k[from_sample:to_sample, :, :, ]],
                                         [X1k[from_sample:to_sample, :, :, ],
                                          X1u[from_sample:to_sample, :, :, ]]],
                                        [y2[from_sample:to_sample, ],
                                         yc[from_sample:to_sample, ]])

        Out = model.predict([X_test_k, X_test_u, X_test_k, X_test_u])
        Acc_v = np.argmax(Out[0], axis=1) - np.argmax(y_test, axis=1)
        acc = (len(Acc_v) - np.count_nonzero(Acc_v) + .0000001) / len(Acc_v)
        logging.info("ACCU: " + str(acc))
        if best_acc < acc:
            best_acc = acc
            logging.info("BEST ACCU: " + str(acc))

    return best_acc