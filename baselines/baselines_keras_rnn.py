import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import keras

from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import GRU
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.layers import Dropout
from data.DataBuilderML400 import DataBuilderML400
from data.DataBuilderPan import DataBuilderPan
from data.base import DataBuilder

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def custom_activation(x):
    return K.max( K.sqrt(x + 1) - 1, 0 )

def rnn_1(data_builder: DataBuilder):
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
    gru_layer = GRU(units=128, activation=custom_activation)

    k_feat = gru_layer(k_embedded_seq)

    u_feat = gru_layer(u_embedded_seq)

    d_layer = Dense(8, activation='relu')
    k_feat = Dense(8, activation='relu')(k_feat)
    u_feat = Dense(8, activation='relu')(u_feat)

    k_feat = keras.layers.Reshape([8, 1])(k_feat)
    u_feat = keras.layers.Reshape([1, 8])(u_feat)
    x = keras.layers.Multiply()([k_feat, u_feat])

    x = Flatten()(x)

    # x = keras.layers.subtract([k_feat, u_feat])

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
    model = rnn_1(ml_data_builder)

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
                                  embed_dim=100, vocab_size=30000, target_doc_len=600, target_sent_len=1024,
                                  word_split=True)
    train_data = data_builder.get_train_data()

    model = rnn_1(data_builder)

    model.fit([np.stack(train_data.value["k_doc"].as_matrix()), np.stack(train_data.value["u_doc"].as_matrix())],
              train_data.label_doc,
              epochs=1, batch_size=32)

    test_data = data_builder.get_test_data()

    loss, acc = model.evaluate(x=[np.stack(test_data.value["k_doc"].as_matrix()),
                                  np.stack(test_data.value["u_doc"].as_matrix())],
                               y=test_data.label_doc, batch_size=32)

    pred_output = model.predict(x=[np.stack(test_data.value["k_doc"].as_matrix()),
                                  np.stack(test_data.value["u_doc"].as_matrix())],
                                batch_size=32)

    logging.info("LOSS: " + str(loss))
    logging.info("ACCU: " + str(acc))

    roc_result = roc_auc_score(test_data.label_doc, pred_output)
    logging.info("ROC: " + str(roc_result))

    get_k_gru_output = K.function(model.input,
                                      [model.get_layer("embedding_layer").get_output_at(0),
                                       model.get_layer("embedding_layer").get_output_at(1)])
    layer_output = get_k_gru_output([test_data.value["k_doc"][0].reshape([1,600]),
                                     test_data.value["u_doc"][0].reshape([1,600])] )
    print(layer_output)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try_pan()
