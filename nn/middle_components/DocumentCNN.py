import tensorflow as tf
import numpy as np


class DocumentCNN(object):
    """
    OVERALL Aspect is one of the aspect, also calculated using attribution method.
    Aspect Class is 5 star class distribution per sentence
    Aspect Class is then scaled using attribution, with throw away (other aspect) allowed.
    (so that's 7 aspect dist: 1 overall, 1 other, 5 normal aspect.)
    """

    def __init__(
            self, previous_component, data,
            filter_size_lists, num_filters,
            dropout=0.0, batch_normalize=False, elu=False, fc=[], l2_reg_lambda=0.0):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.dropout = dropout
        self.batch_normalize = batch_normalize
        self.elu = elu
        self.last_layer = None
        self.l2_reg_lambda = l2_reg_lambda
        self.l2_sum = tf.constant(0.0)

        self.previous_output = previous_component.last_layer

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        num_filters_total = num_filters * len(filter_size_lists[0])

        for i, filter_size in enumerate(filter_size_lists[0]):
            with tf.name_scope("conv-maxpool" + str(i)):
                # Convolution Layer
                filter_shape = [1, filter_size, embedding_size, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.previous_output,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # conv ==> [1, sequence_length - filter_size + 1, 1, 1]
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # h = [batch_size ?, 64, 1022, num_filters]
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, 1, sequence_length - filter_size + 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                # pooled = [batch_size ?, 64, 1, num_filters]

                pooled_outputs.append(pooled)
                # [?, 64, 1, 300]

        # Combine all the pooled features
        self.h_pool = tf.concat(values=pooled_outputs, axis=3)

        # Add dropout
        with tf.name_scope("dropout-keep"):
            # [batch_size * sentence, num_filters_total]
            self.last_layer = tf.nn.dropout(self.h_pool, self.dropout, name="h_drop_sentence")
            # [batch_size, sentence * num_filters_total]
            self.last_layer = tf.reshape(self.last_layer, [-1, document_length, num_filters_total],
                                         name="h_drop_review")

    def get_last_layer_info(self):
        return self.last_layer
