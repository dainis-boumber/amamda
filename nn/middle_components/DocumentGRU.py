import tensorflow as tf
import numpy as np
import logging

from data_helper.Data import DataObject


class DocumentGRU(object):
    """
    OVERALL Aspect is one of the aspect, also calculated using attribution method.
    Aspect Class is 5 star class distribution per sentence
    Aspect Class is then scaled using attribution, with throw away (other aspect) allowed.
    (so that's 7 aspect dist: 1 overall, 1 other, 5 normal aspect.)
    """

    def __init__(
            self, prev_comp, data: DataObject,
            batch_normalize=False, elu=False, hidden_state_dim=128, fc=[]):
        self.dropout = prev_comp.dropout_keep_prob
        self.prev_output = prev_comp.last_layer

        self.document_length = data.target_doc_len
        self.sequence_length = data.target_sent_len
        self.embedding_dim = data.init_embedding.shape[1]

        self.batch_normalize = batch_normalize
        self.elu = elu

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.l2_sum = tf.constant(0.0)

        self.hidden_state_dim = hidden_state_dim
        self.sent_embedding = tf.reshape(self.prev_output, [-1, self.sequence_length, self.embedding_dim],
                                         name="sent_embed")

        self.input_s_len = tf.reshape(prev_comp.input_s_len, [-1], name="s_len_flatten")

        with tf.name_scope("rnn"):
            cell = tf.nn.rnn_cell.GRUCell(self.hidden_state_dim)

            # logging.warning("cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_state_dim)")
            # cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_state_dim)

            logging.warning("DropoutWrapper(cell=cell, output_keep_prob=self.dropout)")
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=self.dropout)

            outputs, encoding = tf.nn.dynamic_rnn(cell=cell, inputs=self.sent_embedding,
                                                  sequence_length=self.input_s_len,
                                                  dtype=tf.float32)
            # encoding = encoding[1]

            # Apply nonlinearity
            # h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Max-pooling over the outputs

        # Add dropout
        with tf.name_scope("dropout-keep"):
            # [batch_size * sentence, num_filters_total]
            # self.last_layer = tf.nn.dropout(encoding, self.dropout, name="h_drop_sentence")
            # [batch_size, sentence * num_filters_total]
            self.last_layer = tf.reshape(encoding, [-1, self.document_length, self.hidden_state_dim],
                                         name="h_drop_review")

    def get_last_layer_info(self):
        return self.last_layer
