import model.ccsa.Initialization as Initialization
import logging
from keras.layers import Lambda
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation, Dropout, Dense

from data.DataBuilderPan import DataBuilderPan

logging.basicConfig(level=logging.INFO)

data_builder = DataBuilderPan(year="15", train_split="pan15_train", test_split="pan15_test",
                       embed_dim=50, vocab_size=30000, target_doc_len=10000, target_sent_len=1024)

model_g = Initialization.dg_cnn(data_builder)

input_ak = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="ak_doc_input")
input_au = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="au_doc_input")
input_bk = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="bk_doc_input")
input_bu = Input(shape=(data_builder.target_doc_len,), dtype='int32', name="bu_doc_input")


# Loss = (1-alpha)Classification_Loss + (alpha)CSA
alpha = .25

# Having two streams. One for source and one for target.
emb_a = model_g([input_ak, input_bk])
emb_b = model_g([input_bk, input_bu])

av_pred = Dense(1, activation='sigmoid')(emb_a)


# Creating the prediction function. This corresponds to h in the paper.
out1 = Dropout(0.5)(processed_a)
out1 = Dense(nb_classes)(out1)
out1 = Activation('softmax', name='classification')(out1)

distance = Lambda(Initialization.euclidean_distance, output_shape=Initialization.eucl_dist_output_shape, name='CSA')(
    [processed_a, processed_b])
model = Model(inputs=[[input_ak, input_au], [input_bk, input_bu]], outputs=[out1, distance])
model.compile(loss={'classification': 'binary_crossentropy', 'CSA': Initialization.contrastive_loss},
              optimizer='adadelta',
              loss_weights={'classification': 1 - alpha, 'CSA': alpha})

tr_pairs, test = Initialization.load_data()
Acc = Initialization.training_the_model(model,tr_pairs, test, epochs=5, batch_size=2)
print(('Best accuracy is {}.'.format(Acc)))
