import tensorflow as tf


class OneSequence:

    def __init__(self, data):
        self.sequence_length = data.sequence_length
        self.num_classes = data.num_classes
        self.vocab_size = data.vocab_size
        self.embedding_size = data.embedding_size
        self.init_embedding = data.init_embedding

        # Placeholders for input, output and dropout, First None is batch size.
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.variable_scope("embedding"):
            if self.init_embedding is None:
                W = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                    name="W")
            else:
                W = tf.Variable(self.init_embedding, name="W", dtype="float32")
            self.embedded = tf.nn.embedding_lookup(W, self.input_x)
            self.last_layer = tf.expand_dims(self.embedded, -1)
