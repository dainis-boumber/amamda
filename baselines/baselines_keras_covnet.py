import logging
import numpy as np
import tensorflow as tf
import keras

from keras.layers import Input
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense

from keras.models import Model

from data_helper.DataBuilderML400 import DataBuilderML400

def try_one():
    vocab_size = 20000
    doc_len = 8192
    ml_data_builder = DataBuilderML400(embed_dim=100, vocab_size=vocab_size,
                                       target_doc_len=doc_len, target_sent_len=1024)
    train_data = ml_data_builder.get_train_data()
    val_data = ml_data_builder.get_val_data()

    embedding_layer = Embedding(input_length=doc_len,
                                input_dim=ml_data_builder.vocabulary_size + 1,
                                output_dim=100,
                                weights=[ml_data_builder.embed_matrix],
                                trainable=False)

    k_input = Input(shape=(doc_len,), dtype='int32', name="k_doc_input")
    k_embedded_seq = embedding_layer(k_input)
    u_input = Input(shape=(doc_len,), dtype='int32', name="u_doc_input")
    u_embedded_seq = embedding_layer(u_input)

    # shared first conv
    conv_first = Conv1D(filters=128, kernel_size=5, activation='relu')
    poll_first = MaxPooling1D(pool_size=1024)

    k_cov = conv_first(k_embedded_seq)
    k_poll = poll_first(k_cov)

    u_cov = conv_first(u_embedded_seq)
    u_poll = poll_first(u_cov)

    x = keras.layers.concatenate([k_poll, u_poll])

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(1, activation='sigmoid')(x)

    model = Model([k_input, u_input], preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    # happy learning!
    model.fit([np.stack(train_data.value["k_doc"].as_matrix()), np.stack(train_data.value["u_doc"].as_matrix())],
              train_data.label_doc,
              validation_data=(
                  [np.stack(val_data.value["k_doc"].as_matrix()),
                   np.stack(val_data.value["u_doc"].as_matrix())],
                  val_data.label_doc),
              epochs=4, batch_size=32)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try_one()