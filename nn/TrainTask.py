import logging
import os
import datetime
import tensorflow as tf
from nn.CNNNetworkBuilder import CNNNetworkBuilder
from data_helper.DataHelpers import DataHelper
import utils.ArchiveManager as AM


class TrainTask:
    """
    This is the MAIN.
    The class set corresponding parameters, log the setting in runs folder.
    the class then create a NN and initialize it.
    lastly the class data batches and feed them into NN for training.
    Currently it only- works with ML data, i'll expand this to be more flexible in the near future.
    """

    def __init__(self, data_helper, am, input_component, middle_component, output_component, batch_size,
                 evaluate_every, checkpoint_every, max_to_keep, restore_path=None):
        self.data_hlp = data_helper
        self.input_comp = input_component
        self.middle_comp = middle_component
        self.output_comp = output_component
        self.am = am

        logging.warning('TrainTask instance initiated: ' + AM.get_date())
        logging.info("Logging to: " + self.am.get_exp_log_path())

        self.exp_dir = self.am.get_exp_dir()

        logging.info("current data is: " + self.data_hlp.problem_name)
        logging.info("current input is: " + type(self.input_comp).__name__)
        logging.info("current middle is: " + type(self.middle_comp).__name__)
        logging.info("current output is: " + type(self.output_comp).__name__)

        self.restore_dir = restore_path
        if restore_path is not None:
            self.restore_latest = tf.train.latest_checkpoint(restore_path + "/checkpoints/")
            logging.warning("RESTORE FROM PATH: " + self.restore_latest)

        # network parameters
        self.batch_size = batch_size
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every
        self.max_to_keep = max_to_keep

        logging.info("setting: %s is %s", "batch_size", self.batch_size)
        logging.info("setting: %s is %s", "evaluate_every", self.evaluate_every)
        logging.info("setting: %s is %s", "checkpoint_every", self.checkpoint_every)

        self.train_data = self.data_hlp.get_train_data()
        self.test_data = self.data_hlp.get_test_data()

        logging.info("Vocabulary Size: {:d}".format(len(self.train_data.vocab)))
        logging.info("Train/Dev split (DOC): {:d}/{:d}".
                     format(len(self.train_data.label_doc), len(self.train_data.label_doc)))
        logging.info("Train/Dev split (IST): {:d}/{:d}".
                     format(len(self.train_data.label_instance), len(self.test_data.label_instance)))

    def generates_all_summary(self, grads_and_vars, graph_loss, graph_acc, graph):
        # Keep track of gradient values and sparsity (optional)
        with tf.name_scope('grad_summary'):
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(":", "_")), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(":", "_")),
                                                         tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        logging.info("Model in {}\n".format(self.am.get_exp_dir()))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", graph_loss)
        acc_summary = tf.summary.scalar("accuracy", graph_acc)

        # Train Summaries
        with tf.name_scope('train_summary'):
            self.train_summary_op = tf.summary.merge(
                [loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(self.exp_dir, "summaries", "train")
            self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, graph)

        # Dev summaries
        with tf.name_scope('dev_summary'):
            self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(self.exp_dir, "summaries", "dev")
            self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, graph)

    def training(self, filter_sizes, num_filters, l2_lambda, dropout_keep_prob,
                 batch_normalize, elu, fc, n_steps):
        logging.info("setting: %s is %s", "filter_sizes", filter_sizes)
        logging.info("setting: %s is %s", "num_filters", num_filters)
        logging.info("setting: %s is %s", "dropout_keep_prob", dropout_keep_prob)
        logging.info("setting: %s is %s", "n_steps", n_steps)
        logging.info("setting: %s is %s", "l2_lambda", l2_lambda)
        logging.info("setting: %s is %s", "batch_normalize", batch_normalize)
        logging.info("setting: %s is %s", "elu", elu)
        logging.info("setting: %s is %s", "fc", fc)

        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess = tf.Session(config=session_conf)
            if self.restore_dir is None:
                cnn = CNNNetworkBuilder(input_comp=self.input_comp,
                                        middle_comp=self.middle_comp,
                                        output_comp=self.output_comp)

                graph_loss = cnn.loss
                graph_accuracy = cnn.accuracy
                graph_input_x = cnn.input_x
                graph_input_y = cnn.input_y
                graph_drop_keep = cnn.dropout_keep_prob

            else:
                saver = tf.train.import_meta_graph("{}.meta".format(self.restore_latest))
                saver.restore(sess, self.restore_latest)

                graph_loss = graph.get_operation_by_name("loss-lbd0/add").outputs[0]
                graph_accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
                graph_input_x = graph.get_operation_by_name("input_x").outputs[0]
                graph_input_y = graph.get_operation_by_name("input_y").outputs[0]
                graph_drop_keep = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                graph_is_train = graph.get_operation_by_name("is_training").outputs[0]

            with sess.as_default():
                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)

                if batch_normalize:
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        optimizer = tf.train.AdamOptimizer(1e-3)
                        grads_and_vars = optimizer.compute_gradients(graph_loss)
                        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                else:
                    optimizer = tf.train.AdamOptimizer(1e-3)
                    grads_and_vars = optimizer.compute_gradients(graph_loss)
                    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                # tf.add_to_collection("optimizer", optimizer)

                self.generates_all_summary(grads_and_vars=grads_and_vars,
                                           graph_loss=graph_loss, graph_acc=graph_accuracy,
                                           graph=sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(self.exp_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=self.max_to_keep)

                # Initialize all variables
                if self.restore_dir is None:
                    sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                feed_dict = {
                    graph_input_x: x_batch,
                    graph_input_y: y_batch,
                    graph_drop_keep: dropout_keep_prob,
                    graph_is_train: 1
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, self.train_summary_op, graph_loss, graph_accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print(("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)))
                if step % 5 == 0:
                    self.train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch):
                feed_dict = {
                    graph_input_x: x_batch,
                    graph_input_y: y_batch,
                    graph_drop_keep: 1,
                    graph_is_train: 0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, self.dev_summary_op, graph_loss, graph_accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print(("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy)))
                self.dev_summary_writer.add_summary(summaries, step)

            # Generate batches
            batches = DataHelper.batch_iter(list(zip(self.train_data.value, self.train_data.label_instance)),
                                            self.batch_size, num_epochs=300)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = list(zip(*batch))
                train_step(x_batch, y_batch)

                current_step = tf.train.global_step(sess, global_step)
                if current_step % self.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_batches = DataHelper.batch_iter(list(zip(self.test_data.value, self.test_data.label_instance)), self.batch_size, 1)
                    for dev_batch in dev_batches:
                        if len(dev_batch) > 0:
                            small_dev_x, small_dev_y = list(zip(*dev_batch))
                            dev_step(small_dev_x, small_dev_y)
                            print("")

                if current_step % self.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print(("Saved model checkpoint to {}\n".format(path)))
                if n_steps is not None and current_step >= n_steps:
                    break
