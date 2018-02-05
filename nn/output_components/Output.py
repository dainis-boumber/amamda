import tensorflow as tf
import logging


class Output(object):
    def __init__(self, prev_layer, data, l2_reg):
        if prev_layer.l2_sum is not None:
            self.l2_sum = prev_layer.l2_sum
            logging.warning("OPTIMIZING PROPER L2")
        else:
            self.l2_sum = tf.constant(0.0)

        # Final (unnormalized) scores and predictions
        with tf.variable_scope("output"):
            # W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            W = tf.get_variable(
                "W",
                shape=[prev_layer.get_shape()[1].value, data.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[data.num_classes]), name="b")

            if l2_reg > 0:
                self.l2_sum += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(prev_layer, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.variable_scope("loss-lbd" + str(l2_reg)):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=self.scores)
            self.loss = tf.reduce_mean(losses) + l2_reg * self.l2_sum
        # Accuracy
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

