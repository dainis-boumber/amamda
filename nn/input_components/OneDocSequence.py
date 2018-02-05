import tensorflow as tf

from data_helper.Data import DataObject


class OneDocSequence:

    def __init__(self, data: DataObject):
        self.document_length = data.target_doc_len
        self.sequence_length = data.target_sent_len
        self.num_aspects = data.num_aspects
        self.num_classes = data.num_classes
        self.vocab_size = len(data.vocab)
        self.embedding_dim = data.init_embedding.shape[1]
        self.init_embedding = data.init_embedding

        # Placeholders for input, output and dropout, First None is batch size.
        self.input_x = tf.placeholder(tf.int32, [None, self.document_length, self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_aspects, self.num_classes], name="input_y")
        self.input_s_len = tf.placeholder(tf.float32, [None, self.document_length], name="input_s_len")
        self.input_s_count = tf.placeholder(tf.float32, [None], name="input_s_count")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.variable_scope("embedding"):
            if self.init_embedding is None:
                W = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0),
                    name="W")
            else:
                W = tf.Variable(self.init_embedding, name="W", dtype="float32", trainable=False)
            self.embedded = tf.nn.embedding_lookup(W, self.input_x, name="embedded_words")
            print(("self.embedded " + str(self.embedded.get_shape())))

            self.last_layer = self.embedded
