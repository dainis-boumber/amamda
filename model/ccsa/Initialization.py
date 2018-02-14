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

def load_data():
    pan_data = PANData(15, 'pan15_train', 'pan15_test')
    train_domains = pan_data.get_train_domains()
    test_gomains = pan_data.get_test_domains()

    return None

def make_pairs(X_source, y_source, X_target, y_target):
    Training_P = []
    Training_N = []

    for trs in range(len(y_source)):
        for trt in range(len(y_target)):
            if y_source[trs] == y_target[trt]:
                Training_P.append([trs, trt])
            else:
                Training_N.append([trs, trt])

    #random.shuffle(Training_N)
    Training = Training_P + Training_N[:3 * len(Training_P)]
    #random.shuffle(Training)

    X1 = np.zeros([len(Training), 16, 16], dtype='float32')
    X2 = np.zeros([len(Training), 16, 16], dtype='float32')

    y1 = np.zeros([len(Training)])
    y2 = np.zeros([len(Training)])
    yc = np.zeros([len(Training)])

    for i in range(len(Training)):
        in1, in2 = Training[i]
        X1[i, :, :] = X_source[in1, :, :]
        X2[i, :, :] = X_target[in2, :, :]

        y1[i] = y_source[in1]
        y2[i] = y_target[in2]
        if y_source[in1] == y_target[in2]:
            yc[i] = 1

    return X1, y1, X2, y2, yc

def training_the_model(model, X1, y1, X2, y2, yc, X_test, y_test, epochs=80, batch_size=256):

    print('Training the model - Epochs '+str(epochs))
    best_acc = 0
    for e in range(epochs):
        if e % 10 == 0:
            printn(str(e) + '->')
        for i in range(len(y2) // batch_size):
            loss = model.train_on_batch([X1[i * batch_size:(i + 1) * batch_size, :, :, :],
                                         X2[i * batch_size:(i + 1) * batch_size, :, :, :]],
                                        [y1[i * batch_size:(i + 1) * batch_size, :],
                                         yc[i * batch_size:(i + 1) * batch_size, ]])
            logging.info("LOSS: " + str(loss))
            loss = model.train_on_batch([X2[i * batch_size:(i + 1) * batch_size, :, :, :],
                                         X1[i * batch_size:(i + 1) * batch_size, :, :, :]],
                                        [y2[i * batch_size:(i + 1) * batch_size, :],
                                         yc[i * batch_size:(i + 1) * batch_size, ]])
            logging.info("LOSS: " + str(loss))

        Out = model.predict([X_test, X_test])
        Acc_v = np.argmax(Out[0], axis=1) - np.argmax(y_test, axis=1)
        acc = (len(Acc_v) - np.count_nonzero(Acc_v) + .0000001) / len(Acc_v)
        logging.info("ACCU: " + str(acc))
        if best_acc < acc:
            best_acc = acc
            logging.info("BEST ACCU: " + str(acc))

    return best_acc

'''
def try_one():
    

    
    if model_save_path.exists():
        model = keras.models.load_model(model_save_path)
    else:
        model = dg_cnn(ml_data_builder)

    model.fit([np.stack(train_data.value["k_doc"].as_matrix()), np.stack(train_data.value["u_doc"].as_matrix())],
              train_data.label_doc,
              validation_data=(
                  [np.stack(val_data.value["k_doc"].as_matrix()),
                   np.stack(val_data.value["u_doc"].as_matrix())],
                  val_data.label_doc),
              epochs=4, batch_size=32)

    test_data = ml_data_builder.get_test_data()

    loss, acc = model.evaluate(x=[np.stack(test_data.value["k_doc"].as_matrix()),
                                  np.stack(test_data.value["u_doc"].as_matrix())],
                               y=test_data.label_doc, batch_size=32)
    
    
'''

