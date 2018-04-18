import model.ccsa.Initialization as Initialization
import logging
from keras.layers import Lambda
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation, Dropout, Dense, Flatten

from data.DataBuilderPan import DataBuilderPan

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    embed_dim = 50
    data_builder = DataBuilderPan(year="15", train_split="pan15_train", test_split="pan15_test",
                                  embed_dim=embed_dim, vocab_size=30000, target_doc_len=10000, target_sent_len=1024)
    train, test = Initialization.load_data(data_builder)

    model_g = Initialization.dg_cnn_yifan(data_builder, embed_dim)

    input_src_k = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="ak_doc_input")
    input_src_u = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="au_doc_input")
    input_tgt_k = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="bk_doc_input")
    input_tgt_u = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="bu_doc_input")

    # Loss = (1-alpha)Classification_Loss + (alpha)CSA
    alpha = 0.25

    # Having two streams. One for source and one for target.
    #emb_src_k, emb_src_u = model_g([input_src_k, input_src_u])
    #emb_tgt_k, emb_tgt_u = model_g([input_tgt_k, input_tgt_u])

    emb_src = model_g([input_src_k, input_src_u])
    emb_tgt = model_g([input_tgt_k, input_tgt_u])
    #diff_embedding = Dense(128, activation='elu')(x)
    # Creating the prediction function. This corresponds to h in the paper.
    #out1 = Dropout(0.5)(emb_a)
    classifier_h = Dense(1, activation='sigmoid', name="av_out")(emb_src)

    CSA_distance = Lambda(Initialization.euclidean_distance,
                      output_shape=Initialization.eucl_dist_output_shape, name='CSA') ([emb_src, emb_tgt])
    model = Model(inputs=[input_src_k, input_src_u, input_tgt_k, input_tgt_u], outputs=[classifier_h, CSA_distance])
    model.compile(loss={'av_out': 'binary_crossentropy', 'CSA': Initialization.contrastive_loss},
                  optimizer='adadelta',
                  loss_weights={'av_out': 1 - alpha, 'CSA': alpha})


    acc = Initialization.training_the_model(model, train, test, epochs=32, batch_size=16)
    print(('Best accuracy is {}.'.format(acc)))
