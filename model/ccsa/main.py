import model.ccsa.Initialization as Initialization
import logging
from keras.layers import Lambda
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation, Dropout, Dense

from data.DataBuilderPan import DataBuilderPan

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data_builder = DataBuilderPan(year="15", train_split="pan15_train", test_split="pan15_test",
                                  embed_dim=100, vocab_size=30000, target_doc_len=10000, target_sent_len=1024)
    train, test = Initialization.load_data(data_builder)

    model_g = Initialization.dg_cnn(data_builder)

    input_ak = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="ak_doc_input")
    input_au = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="au_doc_input")
    input_bk = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="bk_doc_input")
    input_bu = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="bu_doc_input")

    # Loss = (1-alpha)Classification_Loss + (alpha)CSA
    alpha = 0

    # Having two streams. One for source and one for target.
    emb_a = model_g([input_ak, input_au])
    emb_b = model_g([input_bk, input_bu])

    # Creating the prediction function. This corresponds to h in the paper.
    out1 = Dropout(0.5)(emb_a)
    out1 = Dense(1, activation='sigmoid', name="av_out")(out1)

    distance = Lambda(Initialization.euclidean_distance,
                      output_shape=Initialization.eucl_dist_output_shape, name='CSA') ([emb_a, emb_b])
    model = Model(inputs=[input_ak, input_au, input_bk, input_bu], outputs=[out1, distance])
    model.compile(loss={'av_out': 'binary_crossentropy', 'CSA': Initialization.contrastive_loss},
                  optimizer='adadelta',
                  loss_weights={'av_out': 1 - alpha, 'CSA': alpha})


    Acc = Initialization.training_the_model(model, train, test, epochs=10, batch_size=5)
    print(('Best accuracy is {}.'.format(Acc)))
