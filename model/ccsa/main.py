import model.ccsa.Initialization as Initialization
import logging
from keras.layers import Activation, Dropout, Dense
from keras.layers import Lambda
from keras.models import Model
from keras.layers import Input
from keras.layers import Embedding

logging.basicConfig(level=logging.INFO)

doc_len = 1
k_input = Input(shape=(doc_len,), dtype='int32', name="k_doc_input")
u_input = Input(shape=(doc_len,), dtype='int32', name="u_doc_input")
embedding_layer = Embedding(input_length=data_builder.target_doc_len,
                            input_dim=data_builder.vocabulary_size + 1,
                            output_dim=data_builder.embed_dim,
                            weights=[data_builder.embed_matrix],
                            trainable=False)

model_g = Initialization.dg_cnn(k_input=k_input, u_input=u_input, embedding_layer=embedding_layer)

input_shape = (1, 100, 1)
input_ak = Input(shape=input_shape)
input_au = Input(shape=input_shape)
input_bk = Input(shape=input_shape)
input_bu = Input(shape=input_shape)

# number of classes for digits classification
nb_classes = 2

# Loss = (1-alpha)Classification_Loss + (alpha)CSA
alpha = .25

# Having two streams. One for source and one for target.
processed_a = model_g(input_ak, input_au)
processed_b = model_g(input_bk, input_bu)

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
