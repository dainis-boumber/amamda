import logging
from pathlib import Path

import numpy as np
import keras
from keras import backend as K
from keras import regularizers

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import GRU
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.layers import Dropout
from keras.callbacks import Callback
from keras.layers import Lambda

from data.DataBuilderML400 import DataBuilderML400
from data.DataBuilderPan import DataBuilderPan
from data.base import DataBuilder

import matplotlib.pyplot as pyplot


class roc_callback(Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs=None):
        return

    def on_train_end(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        return

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y[0], y_pred[0])
        if "train_roc" in self.model.history.history:
            self.model.history.history["train_roc"].append(roc)
        else:
            self.model.history.history["train_roc"] = [roc]

        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val[0], y_pred_val[0])
        if "val_roc" in self.model.history.history:
            self.model.history.history["val_roc"].append(roc_val)
        else:
            self.model.history.history["val_roc"] = [roc_val]
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs=None):
        return

    def on_batch_end(self, batch, logs=None):
        return

def custom_activation(x):
    return K.max( K.sqrt(x + 1) - 1, 0 )


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


def rnn_concat(data_builder: DataBuilder):
    logging.info("BUILDING RNN USING CONCATENATION")

    embedding_layer = Embedding(input_length=data_builder.target_doc_len,
                                input_dim=data_builder.vocabulary_size + 1,
                                output_dim=100,
                                weights=[data_builder.embed_matrix],
                                trainable=False,
                                mask_zero=True,
                                name="embedding_layer")

    k_input = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="k_doc_input")
    k_embedded_seq = embedding_layer(k_input)
    u_input = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="u_doc_input")
    u_embedded_seq = embedding_layer(u_input)

    # shared first conv
    gru_layer = GRU(units=64, name="gru_layer",
                    dropout=0.3, recurrent_dropout=0.3,
                    reset_after=True)

    k_feat = gru_layer(k_embedded_seq)
    u_feat = gru_layer(u_embedded_seq)

    all_feat = keras.layers.concatenate([k_feat, u_feat])

    # all_feat = Dense(32, activation='relu')(all_feat)
    # all_feat = Dropout(rate=0.3)(all_feat)
    preds = Dense(1, activation='sigmoid', name="av_pred")(all_feat)

    # EMBEDDING DISTANCE
    CSA_distance = Lambda(euclidean_distance, eucl_dist_output_shape, name='doc_dist')\
                        ([k_feat, u_feat])

    model = Model([k_input, u_input], [preds, CSA_distance])

    alpha = 0.6

    model.compile(loss={"av_pred": 'binary_crossentropy', "doc_dist": contrastive_loss},
                  optimizer='adadelta',
                  loss_weights={"av_pred": 1 - alpha, "doc_dist": alpha} )

    return model


def rnn_outer_product(data_builder: DataBuilder):
    logging.info("BUILDING RNN USING OUTER PRODUCT")
    embedding_layer = Embedding(input_length=data_builder.target_doc_len,
                                input_dim=data_builder.vocabulary_size + 1,
                                output_dim=100,
                                weights=[data_builder.embed_matrix],
                                trainable=False,
                                mask_zero=True,
                                name="embedding_layer")

    k_input = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="k_doc_input")
    k_embedded_seq = embedding_layer(k_input)
    u_input = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="u_doc_input")
    u_embedded_seq = embedding_layer(u_input)

    # shared first conv
    gru_layer = GRU(units=64, name="gru_layer",
                    dropout=0.3, recurrent_dropout=0.3)

    k_feat = gru_layer(k_embedded_seq)
    u_feat = gru_layer(u_embedded_seq)

    d_layer = Dense(8, activation='relu')
    k_feat = d_layer(k_feat)
    u_feat = d_layer(u_feat)

    k_feat = keras.layers.Reshape([8, 1])(k_feat)
    u_feat = keras.layers.Reshape([1, 8])(u_feat)
    x = keras.layers.Multiply()([k_feat, u_feat])
    x = Flatten()(x)

    # x = Dense(32, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    model = Model([k_input, u_input], preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['acc'])

    return model


def try_ml():
    ml_data_builder = DataBuilderML400(embed_dim=100, vocab_size=20000,
                                       target_doc_len=8000, target_sent_len=1024)
    train_data = ml_data_builder.get_train_data()
    val_data = ml_data_builder.get_val_data()

    # save_path = Path("temp_model.h5")
    # if save_path.exists():
    #     model = keras.models.load_model(save_path)
    # else:
    model = rnn_outer_product(ml_data_builder)

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
    logging.info("LOSS: " + str(loss))
    logging.info("ACCU: " + str(acc))


def try_pan():
    data_builder = DataBuilderPan(year="15", train_split="pan15_train", test_split="pan15_test",
                                  embed_dim=100, vocab_size=5000, target_doc_len=600, target_sent_len=1024,
                                  word_split=True)
    train_data = data_builder.get_train_data()
    test_data = data_builder.get_test_data()

    model = rnn_concat(data_builder)

    input_x = [np.stack(train_data.value["k_doc"].as_matrix()), np.stack(train_data.value["u_doc"].as_matrix())]
    val_x = [np.stack(test_data.value["k_doc"].as_matrix()),
             np.stack(test_data.value["u_doc"].as_matrix())]
    val_y = test_data.label_doc

    # TRAIN \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    history = model.fit(
        input_x,
        [train_data.label_doc, train_data.label_doc],
        epochs=15, batch_size=32,
        callbacks=[roc_callback(training_data=(input_x, [train_data.label_doc, train_data.label_doc]),
                                validation_data=(val_x, [val_y, val_y]))]
        , validation_data=(val_x, [val_y, val_y])
    )


    # loss, acc = model.evaluate(x=[np.stack(test_data.value["k_doc"].as_matrix()),
    #                               np.stack(test_data.value["u_doc"].as_matrix())],
    #                            y=test_data.label_doc, batch_size=32)

    pred_output = model.predict(x=[np.stack(test_data.value["k_doc"].as_matrix()),
                                  np.stack(test_data.value["u_doc"].as_matrix())],
                                batch_size=32)

    acc = accuracy_score(test_data.label_doc, np.rint(pred_output[0]).astype(int))
    logging.info("ACCU: " + str(acc))

    roc_result = roc_auc_score(test_data.label_doc, pred_output[0])
    logging.info("ROC: " + str(roc_result))

    print("pred: ")
    print(pred_output[0][:10])
    print("dist: ")
    print(pred_output[1][:10])

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()

    pyplot.plot(history.history['train_roc'])
    pyplot.plot(history.history['val_roc'])
    pyplot.title('model train vs validation ROC AUC')
    pyplot.ylabel('AUC')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()

    pyplot.hist(pred_output[0], 20, (0.0, 1.0))
    pyplot.show()

    # get_k_gru_output = K.function(model.input,
    #                                   [model.get_layer("gru_layer").get_output_at(0),
    #                                    model.get_layer("gru_layer").get_output_at(1)])
    # layer_output = get_k_gru_output([test_data.value["k_doc"][0].reshape([1,400]),
    #                                  test_data.value["u_doc"][0].reshape([1,400])] )
    # print(layer_output)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try_pan()
